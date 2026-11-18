import * as THREE from 'three';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { PointsBVH } from '../src/PointsBVH';
import { RaycastPointCloud } from '../src/RaycastPointCloud';
import {
    expectSameHits,
    makePointsNode,
    makePotreeLikeCloud,
    makeRays,
    mulberry32,
    nextTask,
    samplePoints,
} from './helpers';

const THRESHOLDS = [0.4, 0.8, 1.5];

function pair(threshold: number) {
    const stock = new THREE.Raycaster();
    const fast = new RaycastPointCloud();
    stock.params.Points = { threshold };
    fast.params.Points = { threshold };
    return { stock, fast };
}

function compareObject(
    root: THREE.Object3D,
    rays: THREE.Ray[],
    threshold: number,
    configure?: (r: THREE.Raycaster) => void,
): number {
    const { stock, fast } = pair(threshold);
    configure?.(stock);
    configure?.(fast);
    let total = 0;
    for (const ray of rays) {
        stock.ray.copy(ray);
        fast.ray.copy(ray);
        const expected = stock.intersectObject(root, true);
        const actual = fast.intersectObject(root, true);
        expectSameHits(actual, expected);
        total += expected.length;
    }
    return total;
}

afterEach(() => {
    vi.restoreAllMocks();
});

describe('RaycastPointCloud matches THREE.Raycaster exactly', () => {
    it.each(THRESHOLDS)('Potree-like cloud, threshold %s', (threshold) => {
        const spy = vi.spyOn(PointsBVH.prototype, 'collectRayHits');
        const cloud = makePotreeLikeCloud({ seed: 1, nanEvery: 97 });
        const rays = makeRays(mulberry32(2), samplePoints(cloud), 150, 80);
        const total = compareObject(cloud, rays, threshold);
        expect(total).toBeGreaterThan(100);
        expect(spy).toHaveBeenCalled();
    });

    it('rotated, non-uniformly scaled and translated root', () => {
        const cloud = makePotreeLikeCloud({ seed: 3 });
        cloud.rotation.set(0.3, -1.1, 0.7);
        cloud.scale.set(1.5, 0.75, 2);
        cloud.position.set(12, -5, 30);
        cloud.updateMatrixWorld(true);
        const rays = makeRays(mulberry32(4), samplePoints(cloud), 150, 80);
        for (const t of THRESHOLDS) compareObject(cloud, rays, t);
    });

    it('scaled nodes', () => {
        const cloud = makePotreeLikeCloud({ seed: 5 });
        let i = 0;
        cloud.traverse((node) => {
            if (node instanceof THREE.Points && i++ % 2 === 0) {
                node.scale.setScalar(1.7);
            }
        });
        cloud.updateMatrixWorld(true);
        const rays = makeRays(mulberry32(6), samplePoints(cloud), 150, 80);
        for (const t of THRESHOLDS) compareObject(cloud, rays, t);
    });

    it('large world coordinates (mine/UTM scale)', () => {
        const cloud = makePotreeLikeCloud({
            seed: 7,
            worldOffset: new THREE.Vector3(512345.6, 7012345.8, 1234.5),
        });
        const rays = makeRays(mulberry32(8), samplePoints(cloud), 150, 80);
        for (const t of THRESHOLDS) compareObject(cloud, rays, t);
    });

    it('near and far limits', () => {
        const cloud = makePotreeLikeCloud({ seed: 9 });
        const rays = makeRays(mulberry32(10), samplePoints(cloud), 150, 80);
        compareObject(cloud, rays, 0.8, (r) => {
            r.near = 5;
            r.far = 30;
        });
    });

    it('layers: disabled nodes are skipped the same way', () => {
        const cloud = makePotreeLikeCloud({ seed: 11 });
        let i = 0;
        cloud.traverse((node) => {
            if (node instanceof THREE.Points && i++ % 3 === 0) {
                node.layers.set(2);
            }
        });
        const rays = makeRays(mulberry32(12), samplePoints(cloud), 150, 80);
        compareObject(cloud, rays, 0.8);
    });

    it('mixed scene: meshes, indexed and drawRange Points fall back', () => {
        const rng = mulberry32(13);
        const cloud = makePotreeLikeCloud({ seed: 13 });
        const mesh = new THREE.Mesh(
            new THREE.BoxGeometry(10, 10, 10),
            new THREE.MeshBasicMaterial(),
        );
        cloud.children[0].add(mesh);
        const indexed = makePointsNode(rng, 300, 12);
        indexed.geometry.setIndex([...Array(300).keys()]);
        cloud.add(indexed);
        const ranged = makePointsNode(rng, 300, 12);
        ranged.geometry.setDrawRange(10, 100);
        cloud.add(ranged);
        cloud.updateMatrixWorld(true);
        const rays = makeRays(rng, samplePoints(cloud), 150, 80);
        for (const t of THRESHOLDS) compareObject(cloud, rays, t);
    });

    it('intersectObjects over several clouds and objects', () => {
        const a = makePotreeLikeCloud({ seed: 15 });
        const b = makePotreeLikeCloud({
            seed: 16,
            worldOffset: new THREE.Vector3(10, 0, 0),
        });
        const mesh = new THREE.Mesh(
            new THREE.SphereGeometry(8),
            new THREE.MeshBasicMaterial(),
        );
        mesh.updateMatrixWorld(true);
        const objects = [a, mesh, b];
        const rays = makeRays(
            mulberry32(17),
            [...samplePoints(a), ...samplePoints(b)],
            150,
            80,
        );
        for (const recursive of [true, false]) {
            for (const t of THRESHOLDS) {
                const { stock, fast } = pair(t);
                for (const ray of rays) {
                    stock.ray.copy(ray);
                    fast.ray.copy(ray);
                    expectSameHits(
                        fast.intersectObjects(objects, recursive),
                        stock.intersectObjects(objects, recursive),
                    );
                }
            }
        }
    });

    it('appends to a passed-in intersects array like three.js', () => {
        const cloud = makePotreeLikeCloud({ seed: 19 });
        const [ray] = makeRays(mulberry32(20), samplePoints(cloud), 1, 80);
        const { stock, fast } = pair(0.8);
        stock.ray.copy(ray);
        fast.ray.copy(ray);
        const seed = { distance: -1 } as THREE.Intersection;
        const expected = stock.intersectObject(cloud, true, [{ ...seed }]);
        const target = [{ ...seed }];
        const actual = fast.intersectObject(cloud, true, target);
        expect(actual).toBe(target);
        expect(actual.length).toBe(expected.length);
        expectSameHits(actual.slice(1), expected.slice(1));
    });
});

describe('cache invalidation', () => {
    it('edited positions (attribute version) are picked up', async () => {
        const cloud = makePotreeLikeCloud({ seed: 21 });
        const rng = mulberry32(22);
        const rays = makeRays(rng, samplePoints(cloud), 80, 80);
        compareObject(cloud, rays, 0.8);
        await nextTask();

        cloud.traverse((node) => {
            if (!(node instanceof THREE.Points)) return;
            const attr = node.geometry.attributes
                .position as THREE.BufferAttribute;
            for (let i = 0; i < attr.array.length; i++) {
                attr.array[i] += (rng() - 0.5) * 4;
            }
            attr.needsUpdate = true;
            node.geometry.computeBoundingSphere();
        });
        const newRays = makeRays(rng, samplePoints(cloud), 80, 80);
        compareObject(cloud, newRays, 0.8);
    });

    it('moved nodes are picked up in the next task', async () => {
        const cloud = makePotreeLikeCloud({ seed: 23 });
        const rng = mulberry32(24);
        compareObject(cloud, makeRays(rng, samplePoints(cloud), 50, 80), 0.8);
        await nextTask();

        cloud.children.forEach((node) => node.position.addScalar(7));
        cloud.updateMatrixWorld(true);
        compareObject(cloud, makeRays(rng, samplePoints(cloud), 80, 80), 0.8);
    });

    it('a moved root is picked up within the same synchronous run', () => {
        const cloud = makePotreeLikeCloud({ seed: 25 });
        const rng = mulberry32(26);
        compareObject(cloud, makeRays(rng, samplePoints(cloud), 50, 80), 0.8);

        cloud.position.set(40, -20, 15);
        cloud.rotation.y = 0.9;
        cloud.updateMatrixWorld(true);
        compareObject(cloud, makeRays(rng, samplePoints(cloud), 80, 80), 0.8);
    });
});
