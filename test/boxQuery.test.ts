import * as THREE from 'three';
import { describe, expect, it } from 'vitest';

import { RaycastPointCloud } from '../src/RaycastPointCloud';
import { makePotreeLikeCloud } from './helpers';

// The naive pattern getPointsInLocalBox must reproduce exactly.
function naive(
    clouds: THREE.Object3D[],
    localObject: THREE.Object3D,
    box: THREE.Box3,
): THREE.Vector3[] {
    let points: THREE.Vector3[] = [];
    for (const cloud of clouds) {
        cloud.updateMatrixWorld(true);
        cloud.traverse((node) => {
            if (node.type !== 'Points') return;
            const geometry = (node as THREE.Points).geometry;
            if (!geometry || !geometry.attributes.position) return;
            const array = geometry.attributes.position.array;
            for (let i = 0; i < array.length; i += 3) {
                const p = new THREE.Vector3(
                    array[i],
                    array[i + 1],
                    array[i + 2],
                ).applyMatrix4(node.matrixWorld);
                if (!isNaN(p.x) && !isNaN(p.y) && !isNaN(p.z)) points.push(p);
            }
        });
    }
    points.forEach((p) => localObject.worldToLocal(p));
    points = points.filter((p) => !isNaN(p.x) && !isNaN(p.y));
    return points.filter((p) => box.containsPoint(p));
}

describe('RaycastPointCloud.getPointsInLocalBox', () => {
    it.each([
        ['axis-aligned slab', 0, new THREE.Vector3(0, 0, 0)],
        ['rotated slab', 0.8, new THREE.Vector3(5, -3, 2)],
        [
            'far from origin',
            1.3,
            new THREE.Vector3(512345.6, 7012345.8, 1234.5),
        ],
    ])('matches the naive pattern: %s', (_, angle, offset) => {
        const clouds = [
            makePotreeLikeCloud({ seed: 41, worldOffset: offset, nanEvery: 53 }),
            makePotreeLikeCloud({ seed: 42, worldOffset: offset }),
        ];
        const localObject = new THREE.Object3D();
        localObject.position.copy(offset);
        localObject.rotation.set(angle, angle / 2, -angle);
        localObject.updateMatrixWorld(true);
        const box = new THREE.Box3(
            new THREE.Vector3(-40, -40, -1.5),
            new THREE.Vector3(40, 40, 1.5),
        );

        const expected = naive(clouds, localObject, box);
        const actual = RaycastPointCloud.getPointsInLocalBox(
            clouds,
            localObject,
            box,
        );
        expect(expected.length).toBeGreaterThan(50);
        expect(actual).toStrictEqual(expected);
    });

    it('returns nothing when the box misses every node', () => {
        const cloud = makePotreeLikeCloud({ seed: 43 });
        const localObject = new THREE.Object3D();
        const box = new THREE.Box3(
            new THREE.Vector3(1000, 1000, 1000),
            new THREE.Vector3(1001, 1001, 1001),
        );
        expect(
            RaycastPointCloud.getPointsInLocalBox([cloud], localObject, box),
        ).toEqual([]);
    });
});
