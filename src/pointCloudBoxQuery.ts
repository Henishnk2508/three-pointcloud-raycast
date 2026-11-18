import * as THREE from 'three';

import { computeFinitePointBounds } from './PointsBVH';
import type { MaybePoints, PositionArray } from './types';

/**
 * Points of point clouds that fall inside a box given in another object's
 * local space (e.g. a thin slab around a plane).
 *
 * Returns exactly the points this naive pattern keeps, in the same order
 * and with the same values:
 *
 *   points = clouds.flatMap((cloud) => {
 *       cloud.updateMatrixWorld?.(true);
 *       return allWorldPointsOf(cloud); // every Points node, NaN-free
 *   });
 *   points.forEach((p) => localObject.worldToLocal(p));
 *   points = points.filter((p) => !isNaN(p.x) && !isNaN(p.y));
 *   points.filter((p) => localBox.containsPoint(p));
 *
 * That pattern turns every point of every cloud into a Vector3 and inverts
 * localObject's matrix once per point. Here each point is still transformed
 * with the same two matrix applications, but:
 *  - the inverse matrix is computed once (worldToLocal computes the same
 *    inverse for every point),
 *  - only points that end up inside the box are allocated,
 *  - octree nodes whose bounds cannot reach the box are skipped. The bounds
 *    test uses a margin far larger than the rounding of the per point math,
 *    so a skipped node never contains a point the naive pattern would keep.
 */

const _inverse = /* @__PURE__ */ new THREE.Matrix4();
const _nodeToLocal = /* @__PURE__ */ new THREE.Matrix4();
const _point = /* @__PURE__ */ new THREE.Vector3();

interface PositionAttributeLike {
    array: PositionArray;
    version: number;
}

interface BoundsCacheEntry {
    array: PositionArray;
    bounds: number[] | null;
    length: number;
    version: number;
}

// position attribute -> { array, length, version, bounds }
const boundsCache = new WeakMap<PositionAttributeLike, BoundsCacheEntry>();

// Bounds are only needed to skip nodes, so they are cached per attribute and
// revalidated against the array and version.
function getTripletBounds(attribute: PositionAttributeLike): number[] | null {
    const array = attribute.array;
    const cached = boundsCache.get(attribute);
    if (
        cached !== undefined &&
        cached.array === array &&
        cached.length === array.length &&
        cached.version === attribute.version
    ) {
        return cached.bounds;
    }
    const bounds = computeFinitePointBounds(array, array.length / 3);
    boundsCache.set(attribute, {
        array,
        bounds,
        length: array.length,
        version: attribute.version,
    });
    return bounds;
}

function isFiniteAffine(e: number[]): boolean {
    for (let i = 0; i < 16; i++) {
        if (!Number.isFinite(e[i])) return false;
    }
    return e[3] === 0 && e[7] === 0 && e[11] === 0 && e[15] === 1;
}

// True when no point of the node can land inside `box` (localObject space).
function isNodeOutsideBox(
    attribute: PositionAttributeLike,
    nodeMatrixWorld: THREE.Matrix4,
    inverse: THREE.Matrix4,
    box: THREE.Box3,
): boolean {
    const positions = attribute.array;
    // Only plain typed xyz arrays are skipped; anything unusual is processed
    // point by point exactly like the naive pattern.
    if (!ArrayBuffer.isView(positions) || positions.length % 3 !== 0) {
        return false;
    }
    if (!isFiniteAffine(nodeMatrixWorld.elements)) return false;

    const bounds = getTripletBounds(attribute);
    // No finite point: with an affine matrix every point becomes NaN and is
    // dropped by the NaN filters.
    if (bounds === null) return true;

    _nodeToLocal.multiplyMatrices(inverse, nodeMatrixWorld);
    const e = _nodeToLocal.elements;
    for (let i = 0; i < 16; i++) {
        if (!Number.isFinite(e[i])) return false;
    }

    const cx = (bounds[0] + bounds[3]) / 2;
    const cy = (bounds[1] + bounds[4]) / 2;
    const cz = (bounds[2] + bounds[5]) / 2;
    const hx = (bounds[3] - bounds[0]) / 2;
    const hy = (bounds[4] - bounds[1]) / 2;
    const hz = (bounds[5] - bounds[2]) / 2;
    const lx = e[0] * cx + e[4] * cy + e[8] * cz + e[12];
    const ly = e[1] * cx + e[5] * cy + e[9] * cz + e[13];
    const lz = e[2] * cx + e[6] * cy + e[10] * cz + e[14];
    const ex = Math.abs(e[0]) * hx + Math.abs(e[4]) * hy + Math.abs(e[8]) * hz;
    const ey = Math.abs(e[1]) * hx + Math.abs(e[5]) * hy + Math.abs(e[9]) * hz;
    const ez = Math.abs(e[2]) * hx + Math.abs(e[6]) * hy + Math.abs(e[10]) * hz;

    // The naive pattern goes through world coordinates, so its rounding
    // scales with the world magnitude; 1 mm plus a relative term covers it.
    const w = nodeMatrixWorld.elements;
    const margin =
        1e-3 +
        1e-9 *
            (Math.abs(w[12]) +
                Math.abs(w[13]) +
                Math.abs(w[14]) +
                Math.abs(lx) +
                Math.abs(ly) +
                Math.abs(lz) +
                ex +
                ey +
                ez);

    return (
        lx + ex + margin < box.min.x ||
        lx - ex - margin > box.max.x ||
        ly + ey + margin < box.min.y ||
        ly - ey - margin > box.max.y ||
        lz + ez + margin < box.min.z ||
        lz - ez - margin > box.max.z
    );
}

// The per point steps of the naive pattern for one point already in
// `point` (world space). Returns a new Vector3 when it is kept, else null.
function keepIfInBox(
    point: THREE.Vector3,
    inverse: THREE.Matrix4,
    box: THREE.Box3,
): THREE.Vector3 | null {
    if (isNaN(point.x) || isNaN(point.y) || isNaN(point.z)) return null;
    point.applyMatrix4(inverse);
    if (isNaN(point.x) || isNaN(point.y)) return null;
    return box.containsPoint(point) ? point.clone() : null;
}

/**
 * Points of `pointClouds` that fall inside `localBox`, a box in the local
 * space of `localObject`.
 *
 * @param pointClouds clouds to read (Potree or THREE.Points)
 * @param localObject object whose local space `localBox` is in
 * @param localBox box in localObject space
 * @returns kept points in localObject space
 */
export function getPointCloudPointsInLocalBox(
    pointClouds: readonly THREE.Object3D[],
    localObject: THREE.Object3D,
    localBox: THREE.Box3,
): THREE.Vector3[] {
    const result: THREE.Vector3[] = [];
    // Object3D.worldToLocal uses this same inverse for every point.
    _inverse.copy(localObject.matrixWorld).invert();

    for (const pointCloud of pointClouds) {
        pointCloud.updateMatrixWorld?.(true);
        pointCloud.traverse((object) => {
            const node = object as MaybePoints;
            if (node.type !== 'Points') return;
            if (!node.geometry || !node.geometry.attributes.position) return;

            const attribute = node.geometry.attributes
                .position as unknown as PositionAttributeLike;
            const positions = attribute.array;
            const matrixWorld = node.matrixWorld;
            if (isNodeOutsideBox(attribute, matrixWorld, _inverse, localBox)) {
                return;
            }

            const fullLength = positions.length - (positions.length % 3);
            for (let i = 0; i < fullLength; i += 3) {
                _point
                    .set(positions[i], positions[i + 1], positions[i + 2])
                    .applyMatrix4(matrixWorld);
                const kept = keepIfInBox(_point, _inverse, localBox);
                if (kept !== null) result.push(kept);
            }
            if (fullLength < positions.length) {
                // A trailing partial triplet, built the way the naive pattern
                // builds every point (missing components default to 0).
                const point = new THREE.Vector3(
                    positions[fullLength],
                    positions[fullLength + 1],
                    positions[fullLength + 2],
                ).applyMatrix4(matrixWorld);
                const kept = keepIfInBox(point, _inverse, localBox);
                if (kept !== null) result.push(kept);
            }
        });
    }

    return result;
}
