import type * as THREE from 'three';

/** Flat xyz triplets, e.g. `BufferAttribute.array`. */
export type PositionArray = ArrayLike<number>;

/**
 * The parts of a Potree `PointCloudOctree` this library reads. Potree is
 * detected by shape, so no Potree import or type package is needed.
 */
export interface PotreeLikeCloud extends THREE.Object3D {
    pcoGeometry: unknown;
    root?: { sceneNode?: THREE.Object3D | null } | null;
}

/** Options for {@link PointsBVH}. */
export interface PointsBVHOptions {
    /** Maximum points per leaf. Default 32. */
    maxLeafSize?: number;
    /** Maximum tree depth. Default 40. */
    maxDepth?: number;
}

/** Output buffer of {@link PointsBVH.collectRayHits}; grown when needed. */
export interface HitIndexBuffer {
    array: Uint32Array;
}

/** An Object3D that may be a THREE.Points (or a Potree octree node). */
export interface MaybePoints extends THREE.Object3D {
    isPoints?: boolean;
    geometry?: THREE.BufferGeometry;
}
