import * as THREE from 'three';
import { expect } from 'vitest';

/** Deterministic PRNG so failures are reproducible. */
export function mulberry32(seed: number): () => number {
    let a = seed >>> 0;
    return () => {
        a = (a + 0x6d2b79f5) >>> 0;
        let t = a;
        t = Math.imul(t ^ (t >>> 15), t | 1);
        t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
}

export interface FakeCloudOptions {
    seed: number;
    nodeCount?: number;
    pointsPerNode?: number;
    nodeSpread?: number;
    nodeSize?: number;
    worldOffset?: THREE.Vector3;
    nanEvery?: number;
}

export function makePointsNode(
    rng: () => number,
    count: number,
    size: number,
    nanEvery = 0,
): THREE.Points {
    const positions = new Float32Array(count * 3);
    for (let i = 0; i < count; i++) {
        positions[i * 3] = (rng() - 0.5) * size;
        positions[i * 3 + 1] = (rng() - 0.5) * size;
        positions[i * 3 + 2] = (rng() - 0.5) * size;
        if (nanEvery > 0 && i % nanEvery === nanEvery - 1) {
            positions[i * 3 + 1] = NaN;
        }
    }
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    if (nanEvery > 0) {
        // three.js would compute a NaN sphere and reject every ray, so set
        // one from the finite points, as a Potree loader would.
        const finite: THREE.Vector3[] = [];
        for (let i = 0; i < count; i++) {
            const p = new THREE.Vector3().fromArray(positions, i * 3);
            if (Number.isFinite(p.x + p.y + p.z)) finite.push(p);
        }
        geometry.boundingSphere = new THREE.Sphere().setFromPoints(finite);
    }
    return new THREE.Points(geometry, new THREE.PointsMaterial());
}

/**
 * An object shaped like a Potree PointCloudOctree: a root with `pcoGeometry`
 * and `root.sceneNode`, holding a nested hierarchy of THREE.Points nodes.
 */
export function makePotreeLikeCloud(options: FakeCloudOptions): THREE.Object3D {
    const {
        seed,
        nodeCount = 24,
        pointsPerNode = 600,
        nodeSpread = 60,
        nodeSize = 12,
        worldOffset,
        nanEvery = 0,
    } = options;
    const rng = mulberry32(seed);
    const root = new THREE.Object3D();
    if (worldOffset) root.position.copy(worldOffset);

    const nodes: THREE.Points[] = [];
    for (let i = 0; i < nodeCount; i++) {
        const node = makePointsNode(rng, pointsPerNode, nodeSize, nanEvery);
        node.position.set(
            (rng() - 0.5) * nodeSpread,
            (rng() - 0.5) * nodeSpread,
            (rng() - 0.5) * nodeSpread,
        );
        // Nest some nodes like an octree: parent is a random earlier node.
        const parent = i === 0 || rng() < 0.4 ? root : nodes[(rng() * i) | 0];
        if (parent !== root) node.position.sub(parent.position);
        parent.add(node);
        nodes.push(node);
    }

    Object.assign(root, {
        pcoGeometry: {},
        root: { sceneNode: nodes[0] },
    });
    root.updateMatrixWorld(true);
    return root;
}

/** Collects every THREE.Points world-space point of `object` (finite only). */
export function samplePoints(object: THREE.Object3D): THREE.Vector3[] {
    const out: THREE.Vector3[] = [];
    object.updateMatrixWorld(true);
    object.traverse((node) => {
        if (!(node instanceof THREE.Points)) return;
        const attr = node.geometry.attributes.position as THREE.BufferAttribute;
        for (let i = 0; i < attr.count; i++) {
            const p = new THREE.Vector3()
                .fromBufferAttribute(attr, i)
                .applyMatrix4(node.matrixWorld);
            if (Number.isFinite(p.x + p.y + p.z)) out.push(p);
        }
    });
    return out;
}

/** Rays aimed near cloud points, so most of them hit something. */
export function makeRays(
    rng: () => number,
    targets: THREE.Vector3[],
    count: number,
    originSpread: number,
): THREE.Ray[] {
    const rays: THREE.Ray[] = [];
    for (let i = 0; i < count; i++) {
        const target = targets[(rng() * targets.length) | 0].clone();
        target.x += (rng() - 0.5) * 2;
        target.y += (rng() - 0.5) * 2;
        target.z += (rng() - 0.5) * 2;
        const origin = target
            .clone()
            .add(
                new THREE.Vector3(
                    rng() - 0.5,
                    rng() - 0.5,
                    rng() - 0.5,
                ).multiplyScalar(originSpread),
            );
        const direction = target.clone().sub(origin).normalize();
        rays.push(new THREE.Ray(origin, direction));
    }
    return rays;
}

function snapshot(hits: THREE.Intersection[]) {
    return hits.map((hit) => ({
        ...hit,
        keys: Object.keys(hit).join(','),
        object: hit.object.id,
    }));
}

/** Asserts two hit arrays are identical: values, order, key order, objects. */
export function expectSameHits(
    actual: THREE.Intersection[],
    expected: THREE.Intersection[],
): void {
    expect(snapshot(actual)).toStrictEqual(snapshot(expected));
    for (let i = 0; i < actual.length; i++) {
        expect(actual[i].object).toBe(expected[i].object);
    }
}

export function nextTask(): Promise<void> {
    return new Promise((resolve) => setTimeout(resolve, 0));
}
