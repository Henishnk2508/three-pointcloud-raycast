import * as THREE from 'three';

import { computeFinitePointBounds, PointsBVH } from './PointsBVH';
import type {
    HitIndexBuffer,
    MaybePoints,
    PositionArray,
    PotreeLikeCloud,
} from './types';

/**
 * Accelerated raycasting for Potree point clouds.
 *
 * `intersectPointCloud(raycaster, cloud)` returns exactly what
 * `raycaster.intersectObject(cloud, true)` returns: the same hits with the
 * same values, in the same order. It just finds them faster.
 *
 * Why the stock raycast is slow: three.js visits every octree node of the
 * cloud and, for each node whose bounding sphere touches the ray, tests every
 * point of that node. A ray crossing a 1.5M point cloud tests hundreds of
 * thousands of points.
 *
 * How this is faster (a two level BVH, the idea behind three-mesh-bvh):
 *  - Top level: a BVH over the world bounds of the cloud's octree nodes. It
 *    finds the few nodes the ray can reach. Rebuilt once per JS task (cheap).
 *  - Bottom level: a PointsBVH per node geometry, in local space. It finds
 *    the few points near the ray. Built lazily, cached per geometry.
 *
 * Why the results are identical:
 *  - Both BVH levels only prune. Their boxes are grown by the pick radius plus
 *    a safety margin, so a point that three.js would hit is never pruned.
 *  - Every surviving point goes through the same arithmetic as three.js
 *    Points.raycast/testPoint (bounding sphere check, local ray, squared
 *    distance, closest point, world transform, near/far), so the hit values
 *    are bit-for-bit the same.
 *  - Hits are pushed in the order three.js would push them (scene graph
 *    depth-first order, then ascending point index), and then sorted with
 *    the same stable comparator, so the final array order is the same too.
 *  - Anything that is not a plain Potree octree node (other object types,
 *    indexed or interleaved geometry, custom raycast, ...) is raycast with its
 *    own `raycast` method, exactly like three.js does.
 *
 * Cache validity: Potree only adds, removes or moves octree nodes inside its
 * render and load callbacks, never in the middle of synchronous code. The
 * top level index is therefore reused only within one synchronous run,
 * dropped when that run ends, and also rebuilt if the cloud's own matrix
 * changes. Bottom level trees depend only on a geometry's positions, which
 * Potree never edits after loading, are revalidated through the attribute
 * version on every use and are freed together with their geometry.
 */

type Intersection = THREE.Intersection<THREE.Object3D>;
type RaycastFn = THREE.Object3D['raycast'];

interface GeometryEntry {
    array: PositionArray;
    attribute: THREE.BufferAttribute;
    bounds: number[] | null;
    bvh: PointsBVH | null;
    count: number;
    version: number;
}

const _sphere = /* @__PURE__ */ new THREE.Sphere();
const _inverseMatrix = /* @__PURE__ */ new THREE.Matrix4();
const _ray = /* @__PURE__ */ new THREE.Ray();

// Raycast functions of Potree's octree node class (Potree bundles its own
// three.js, so its Points class differs from the app's THREE.Points).
const potreePointsRaycasts = new Set<RaycastFn>();

// geometry -> { attribute, array, count, version, bounds, bvh }
const geometryCache = new WeakMap<THREE.BufferGeometry, GeometryEntry>();
// cloud root -> CloudIndex, for the current synchronous run only.
const cloudCache = new Map<THREE.Object3D, CloudIndex>();

const hitIndexScratch: HitIndexBuffer = { array: new Uint32Array(256) };

let cloudCacheClearQueued = false;
const scheduleMicrotask: (callback: () => void) => void =
    typeof queueMicrotask === 'function'
        ? queueMicrotask
        : (callback) => {
              Promise.resolve().then(callback);
          };

function clearCloudCache(): void {
    cloudCache.clear();
    cloudCacheClearQueued = false;
}

// Drop every top level index once the current synchronous run finishes,
// which is the earliest moment Potree can change its scene nodes. This also
// releases the node references they hold.
function queueCloudCacheClear(): void {
    if (!cloudCacheClearQueued) {
        cloudCacheClearQueued = true;
        scheduleMicrotask(clearCloudCache);
    }
}

function ascSort(a: Intersection, b: Intersection): number {
    return a.distance - b.distance;
}

/**
 * Drop-in replacement for `raycaster.intersectObject(object, recursive)`.
 * Returns the same array three.js would return, built faster for Potree
 * point clouds.
 *
 * @param raycaster any THREE.Raycaster (its ray, params, layers, near, far)
 * @param object point cloud (or any object)
 * @param recursive also test the descendants
 * @param intersects array the hits are appended to
 * @returns intersections sorted by distance
 */
export function intersectPointCloud<
    TIntersected extends THREE.Object3D = THREE.Object3D,
>(
    raycaster: THREE.Raycaster,
    object: THREE.Object3D,
    recursive = true,
    intersects: THREE.Intersection<TIntersected>[] = [],
): THREE.Intersection<TIntersected>[] {
    const target = intersects as unknown as Intersection[];
    intersectRoot(raycaster, object, recursive, target);
    target.sort(ascSort);
    return intersects;
}

/**
 * Drop-in replacement for `raycaster.intersectObjects(objects, recursive)`.
 *
 * @param raycaster any THREE.Raycaster (its ray, params, layers, near, far)
 * @param objects point clouds (or any objects)
 * @param recursive also test the descendants
 * @param intersects array the hits are appended to
 * @returns intersections sorted by distance
 */
export function intersectPointClouds<
    TIntersected extends THREE.Object3D = THREE.Object3D,
>(
    raycaster: THREE.Raycaster,
    objects: readonly THREE.Object3D[],
    recursive = true,
    intersects: THREE.Intersection<TIntersected>[] = [],
): THREE.Intersection<TIntersected>[] {
    const target = intersects as unknown as Intersection[];
    for (let i = 0, l = objects.length; i < l; i++) {
        intersectRoot(raycaster, objects[i], recursive, target);
    }
    target.sort(ascSort);
    return intersects;
}

function intersectRoot(
    raycaster: THREE.Raycaster,
    object: THREE.Object3D,
    recursive: boolean,
    intersects: Intersection[],
): void {
    // A missing object goes the generic way, so it fails exactly like
    // three.js does (reading `layers` of undefined).
    if (object && recursive === true && isAcceleratableRaycaster(raycaster)) {
        const index = getCloudIndex(object);
        if (index !== null) {
            index.intersect(raycaster, intersects);
            return;
        }
    }
    intersectGeneric(object, raycaster, intersects, recursive);
}

// Same traversal as three.js Raycaster (private intersectObject), without
// the final sort.
function intersectGeneric(
    object: THREE.Object3D,
    raycaster: THREE.Raycaster,
    intersects: Intersection[],
    recursive: boolean,
): void {
    if (object.layers.test(raycaster.layers)) {
        object.raycast(raycaster, intersects);
    }
    if (recursive === true) {
        const children = object.children;
        for (let i = 0, l = children.length; i < l; i++) {
            intersectGeneric(children[i], raycaster, intersects, true);
        }
    }
}

function isAcceleratableRaycaster(raycaster: THREE.Raycaster): boolean {
    const points = raycaster.params && raycaster.params.Points;
    if (!points) return false;
    const threshold: unknown = points.threshold;
    if (typeof threshold !== 'number' || !(threshold >= 0)) return false;
    if (!Number.isFinite(threshold)) return false;
    const { origin, direction } = raycaster.ray;
    if (
        !Number.isFinite(origin.x) ||
        !Number.isFinite(origin.y) ||
        !Number.isFinite(origin.z) ||
        !Number.isFinite(direction.x) ||
        !Number.isFinite(direction.y) ||
        !Number.isFinite(direction.z)
    ) {
        return false;
    }
    if (direction.x === 0 && direction.y === 0 && direction.z === 0) {
        return false;
    }
    const { near, far } = raycaster;
    return (
        typeof near === 'number' &&
        typeof far === 'number' &&
        near === near &&
        far === far
    );
}

function getCloudIndex(root: THREE.Object3D): CloudIndex | null {
    queueCloudCacheClear();
    let index = cloudCache.get(root);
    if (index === undefined || !index.isRootMatrixUnchanged()) {
        index = new CloudIndex(root);
        cloudCache.set(root, index);
    }
    return index.usable ? index : null;
}

// Learn the raycast function of Potree's octree node class from a loaded
// Potree point cloud (PointCloudOctree: has pcoGeometry and root.sceneNode).
function registerPotreeCloud(root: THREE.Object3D): void {
    const cloud = root as Partial<PotreeLikeCloud>;
    const sceneNode = (
        cloud.pcoGeometry ? cloud.root?.sceneNode : null
    ) as MaybePoints | null | undefined;
    if (!sceneNode || sceneNode.isPoints !== true) return;
    const raycast = (Object.getPrototypeOf(sceneNode) as THREE.Object3D)
        .raycast;
    if (typeof raycast === 'function') potreePointsRaycasts.add(raycast);
}

// Cached local bounds and lazily built PointsBVH for a geometry, or null if
// the geometry is not a plain non-indexed xyz point buffer.
function getGeometryEntry(object: MaybePoints): GeometryEntry | null {
    const geometry = object.geometry;
    if (!geometry || geometry.isBufferGeometry !== true) return null;
    if (geometry.index !== null) return null;
    const attribute = (geometry.attributes &&
        geometry.attributes.position) as
        | (THREE.BufferAttribute & { isInterleavedBufferAttribute?: boolean })
        | undefined;
    if (
        !attribute ||
        attribute.isInterleavedBufferAttribute === true ||
        attribute.itemSize !== 3 ||
        attribute.normalized === true ||
        !ArrayBuffer.isView(attribute.array) ||
        attribute.array instanceof BigInt64Array ||
        attribute.array instanceof BigUint64Array ||
        attribute.array.length < attribute.count * 3
    ) {
        return null;
    }
    // three r161 Points.raycast honours drawRange while r124 ignores it. Only
    // the default (full) range is accelerated, where both behave the same.
    const drawRange = geometry.drawRange;
    if (
        drawRange &&
        (drawRange.start !== 0 || drawRange.count < attribute.count)
    ) {
        return null;
    }

    let entry = geometryCache.get(geometry);
    if (
        entry !== undefined &&
        entry.attribute === attribute &&
        entry.array === attribute.array &&
        entry.count === attribute.count &&
        entry.version === attribute.version
    ) {
        return entry;
    }

    entry = {
        array: attribute.array,
        attribute,
        bounds: computeFinitePointBounds(attribute.array, attribute.count),
        bvh: null,
        count: attribute.count,
        version: attribute.version,
    };
    geometryCache.set(geometry, entry);
    return entry;
}

/**
 * Upper bound of how far apart two world points can be when their local
 * points are 1 apart: the spectral norm of the matrix's linear part (exact
 * for rotation/uniform scale, Frobenius norm bound otherwise).
 */
function getLinearNormBound(e: number[]): number {
    const c0 = Math.hypot(e[0], e[1], e[2]);
    const c1 = Math.hypot(e[4], e[5], e[6]);
    const c2 = Math.hypot(e[8], e[9], e[10]);
    const d01 = Math.abs(e[0] * e[4] + e[1] * e[5] + e[2] * e[6]);
    const d02 = Math.abs(e[0] * e[8] + e[1] * e[9] + e[2] * e[10]);
    const d12 = Math.abs(e[4] * e[8] + e[5] * e[9] + e[6] * e[10]);
    const tolerance = 1e-12;
    if (
        d01 <= tolerance * c0 * c1 &&
        d02 <= tolerance * c0 * c2 &&
        d12 <= tolerance * c1 * c2
    ) {
        return Math.max(c0, c1, c2) * (1 + 1e-9);
    }
    return Math.hypot(c0, c1, c2) * (1 + 1e-9);
}

function isUsableMatrix(e: number[]): boolean {
    for (let i = 0; i < 16; i++) {
        if (!Number.isFinite(e[i])) return false;
    }
    // Potree nodes are affine. Anything projective goes the generic way.
    if (e[3] !== 0 || e[7] !== 0 || e[11] !== 0 || e[15] !== 1) return false;
    const det =
        e[0] * (e[5] * e[10] - e[6] * e[9]) -
        e[4] * (e[1] * e[10] - e[2] * e[9]) +
        e[8] * (e[1] * e[6] - e[2] * e[5]);
    return det !== 0 && Number.isFinite(det);
}

class CloudIndex {
    root: THREE.Object3D;
    rootMatrix: number[];
    usable: boolean;
    objects!: THREE.Object3D[];
    genericOrder!: Int32Array;
    primOrder!: Int32Array;
    primEntries!: GeometryEntry[];
    maxRadiusScale!: number;
    tlas!: BoxBVH;

    constructor(root: THREE.Object3D) {
        this.root = root;
        this.rootMatrix = Array.from(root.matrixWorld.elements);
        this.usable = false;

        registerPotreeCloud(root);
        if (potreePointsRaycasts.size === 0) return;

        // Scene graph in the order three.js visits it (depth-first preorder).
        const objects: THREE.Object3D[] = [];
        (function collect(object: THREE.Object3D) {
            objects.push(object);
            const children = object.children;
            for (let i = 0, l = children.length; i < l; i++) {
                collect(children[i]);
            }
        })(root);

        const genericOrder: number[] = [];
        const primOrder: number[] = [];
        const primEntries: GeometryEntry[] = [];
        const primBounds: number[] = [];
        let maxRadiusScale = 0;

        for (let i = 0; i < objects.length; i++) {
            const object = objects[i];
            const entry = this._getAcceleratedEntry(object);
            if (entry === null) {
                genericOrder.push(i);
                continue;
            }
            // A node without a single finite point can never be hit.
            if (entry.bounds === null) continue;

            const e = object.matrixWorld.elements;
            const scale = object.scale;
            const averageScale = (scale.x + scale.y + scale.z) / 3;
            if (!(averageScale > 0) || !Number.isFinite(averageScale)) {
                genericOrder.push(i);
                continue;
            }

            // World AABB of the local bounds (affine: centre + |A| * extent).
            const b = entry.bounds;
            const cx = (b[0] + b[3]) / 2;
            const cy = (b[1] + b[4]) / 2;
            const cz = (b[2] + b[5]) / 2;
            const hx = (b[3] - b[0]) / 2;
            const hy = (b[4] - b[1]) / 2;
            const hz = (b[5] - b[2]) / 2;
            const wx = e[0] * cx + e[4] * cy + e[8] * cz + e[12];
            const wy = e[1] * cx + e[5] * cy + e[9] * cz + e[13];
            const wz = e[2] * cx + e[6] * cy + e[10] * cz + e[14];
            const ex =
                Math.abs(e[0]) * hx + Math.abs(e[4]) * hy + Math.abs(e[8]) * hz;
            const ey =
                Math.abs(e[1]) * hx + Math.abs(e[5]) * hy + Math.abs(e[9]) * hz;
            const ez =
                Math.abs(e[2]) * hx +
                Math.abs(e[6]) * hy +
                Math.abs(e[10]) * hz;
            // Small relative pad for the rounding of this transform.
            const pad =
                1e-9 *
                (Math.abs(wx) + Math.abs(wy) + Math.abs(wz) + ex + ey + ez);
            primBounds.push(
                wx - ex - pad,
                wy - ey - pad,
                wz - ez - pad,
                wx + ex + pad,
                wy + ey + pad,
                wz + ez + pad,
            );
            primOrder.push(i);
            primEntries.push(entry);

            // A hit point is within threshold / averageScale of the local ray,
            // so within that times the matrix norm of the world ray.
            const radiusScale = getLinearNormBound(e) / averageScale;
            if (radiusScale > maxRadiusScale) maxRadiusScale = radiusScale;
        }

        if (primOrder.length === 0) return;

        this.objects = objects;
        this.genericOrder = Int32Array.from(genericOrder);
        this.primOrder = Int32Array.from(primOrder);
        this.primEntries = primEntries;
        this.maxRadiusScale = maxRadiusScale;
        this.tlas = new BoxBVH(Float64Array.from(primBounds), primOrder.length);
        this.usable = true;
    }

    private _getAcceleratedEntry(object: MaybePoints): GeometryEntry | null {
        if (object.isPoints !== true) return null;
        if (!potreePointsRaycasts.has(object.raycast)) return null;
        if (!isUsableMatrix(object.matrixWorld.elements)) return null;
        return getGeometryEntry(object);
    }

    isRootMatrixUnchanged(): boolean {
        const current = this.root.matrixWorld.elements;
        const saved = this.rootMatrix;
        for (let i = 0; i < 16; i++) {
            if (current[i] !== saved[i]) return false;
        }
        return true;
    }

    intersect(raycaster: THREE.Raycaster, intersects: Intersection[]): void {
        const ray = raycaster.ray;
        const ox = ray.origin.x;
        const oy = ray.origin.y;
        const oz = ray.origin.z;
        let dx = ray.direction.x;
        let dy = ray.direction.y;
        let dz = ray.direction.z;
        const length = Math.sqrt(dx * dx + dy * dy + dz * dz);
        dx /= length;
        dy /= length;
        dz /= length;

        const threshold = raycaster.params.Points.threshold;
        const far = raycaster.far;
        const margin =
            1e-6 + 1e-8 * (Math.abs(ox) + Math.abs(oy) + Math.abs(oz));
        const radius = threshold * this.maxRadiusScale + margin;
        const tMax = far === Infinity ? Infinity : far * (1 + 1e-9) + margin;

        // Nodes whose grown world box the ray reaches.
        const candidateCount = this.tlas.collect(
            ox,
            oy,
            oz,
            dx,
            dy,
            dz,
            radius,
            tMax,
        );
        // Prims were created in depth-first order, so sorting prim ids
        // restores the order three.js visits these nodes in.
        const sortedPrims = this.tlas.result.subarray(0, candidateCount);
        sortedPrims.sort();
        const primOrder = this.primOrder;

        // Merge with the objects raycast the generic way, keeping depth-first
        // order across both lists.
        const objects = this.objects;
        const genericOrder = this.genericOrder;
        const genericCount = genericOrder.length;
        let g = 0;
        let c = 0;
        while (g < genericCount || c < candidateCount) {
            const nextGeneric = g < genericCount ? genericOrder[g] : Infinity;
            const nextPrim =
                c < candidateCount ? primOrder[sortedPrims[c]] : Infinity;
            if (nextGeneric < nextPrim) {
                const object = objects[nextGeneric];
                if (object.layers.test(raycaster.layers)) {
                    object.raycast(raycaster, intersects);
                }
                g++;
            } else {
                const object = objects[nextPrim] as THREE.Points;
                if (object.layers.test(raycaster.layers)) {
                    raycastPotreeNode(
                        object,
                        this.primEntries[sortedPrims[c]],
                        raycaster,
                        threshold,
                        intersects,
                    );
                }
                c++;
            }
        }
    }
}

/**
 * three.js Points.raycast for one Potree octree node, with the point loop
 * replaced by a PointsBVH query. Every step that decides a hit or computes a
 * returned value is the same three.js call or the same arithmetic.
 */
function raycastPotreeNode(
    object: THREE.Points,
    entry: GeometryEntry,
    raycaster: THREE.Raycaster,
    threshold: number,
    intersects: Intersection[],
): void {
    const geometry = object.geometry;
    const matrixWorld = object.matrixWorld;

    // Checking boundingSphere distance to ray (as Points.raycast does)
    if (geometry.boundingSphere === null) geometry.computeBoundingSphere();
    _sphere.copy(geometry.boundingSphere!);
    _sphere.applyMatrix4(matrixWorld);
    _sphere.radius += threshold;
    if (raycaster.ray.intersectsSphere(_sphere) === false) return;

    _inverseMatrix.copy(matrixWorld).invert();
    _ray.copy(raycaster.ray).applyMatrix4(_inverseMatrix);

    const localThreshold =
        threshold / ((object.scale.x + object.scale.y + object.scale.z) / 3);
    const localThresholdSq = localThreshold * localThreshold;

    const ox = _ray.origin.x;
    const oy = _ray.origin.y;
    const oz = _ray.origin.z;
    const dx = _ray.direction.x;
    const dy = _ray.direction.y;
    const dz = _ray.direction.z;

    const localMargin =
        1e-6 +
        1e-8 * (Math.abs(ox) + Math.abs(oy) + Math.abs(oz)) +
        1e-9 * localThreshold;

    // A hit's world distance is (local ray parameter) * |A * localDir|.
    let tMaxLocal = Infinity;
    const far = raycaster.far;
    if (far !== Infinity) {
        const e = matrixWorld.elements;
        const sx = e[0] * dx + e[4] * dy + e[8] * dz;
        const sy = e[1] * dx + e[5] * dy + e[9] * dz;
        const sz = e[2] * dx + e[6] * dy + e[10] * dz;
        const stretch = Math.sqrt(sx * sx + sy * sy + sz * sz);
        if (stretch > 0) {
            tMaxLocal = (far / stretch) * (1 + 1e-9) + localMargin;
        }
    }

    if (entry.bvh === null) {
        entry.bvh = new PointsBVH(entry.array, entry.count);
    }
    const hitCount = entry.bvh.collectRayHits(
        ox,
        oy,
        oz,
        dx,
        dy,
        dz,
        localThreshold + localMargin,
        tMaxLocal,
        localThresholdSq,
        hitIndexScratch,
    );
    if (hitCount === 0) return;

    // Points.raycast tests points in index order.
    const hitIndices = hitIndexScratch.array.subarray(0, hitCount);
    hitIndices.sort();

    const positions = entry.array;
    // Same Vector3 class the object's own three.js would create.
    const PointVector = object.position.constructor as new () => THREE.Vector3;
    const rayOrigin = raycaster.ray.origin;
    const near = raycaster.near;

    for (let k = 0; k < hitCount; k++) {
        const index = hitIndices[k];
        const p = 3 * index;
        const px = positions[p];
        const py = positions[p + 1];
        const pz = positions[p + 2];

        // testPoint: _ray.distanceSqToPoint(point)
        const directionDistance =
            (px - ox) * dx + (py - oy) * dy + (pz - oz) * dz;
        let rayPointDistanceSq: number;
        // testPoint: _ray.closestPointToPoint(point, intersectPoint)
        const intersectPoint = new PointVector();
        if (directionDistance < 0) {
            const ex = ox - px;
            const ey = oy - py;
            const ez = oz - pz;
            rayPointDistanceSq = ex * ex + ey * ey + ez * ez;
            intersectPoint.set(ox, oy, oz);
        } else {
            const qx = dx * directionDistance + ox;
            const qy = dy * directionDistance + oy;
            const qz = dz * directionDistance + oz;
            const fx = qx - px;
            const fy = qy - py;
            const fz = qz - pz;
            rayPointDistanceSq = fx * fx + fy * fy + fz * fz;
            intersectPoint.set(qx, qy, qz);
        }
        if (!(rayPointDistanceSq < localThresholdSq)) continue;

        intersectPoint.applyMatrix4(matrixWorld);
        const distance = rayOrigin.distanceTo(intersectPoint);
        if (distance < near || distance > far) continue;

        // Same object shape and key order as three.js testPoint.
        intersects.push({
            distance: distance,
            distanceToRay: Math.sqrt(rayPointDistanceSq),
            point: intersectPoint,
            index: index,
            face: null,
            object: object,
        });
    }
}

/**
 * BVH over axis aligned boxes (the octree nodes of one cloud). Same build and
 * layout as PointsBVH, with boxes as primitives and their centres used for
 * splitting.
 */
class BoxBVH {
    primBounds: Float64Array;
    result: Int32Array;
    prims!: Int32Array;
    bounds!: Float64Array;
    offsetOrRight!: Int32Array;
    leafCounts!: Int32Array;
    stack!: Int32Array;

    constructor(primBounds: Float64Array, primCount: number, maxLeafSize = 4) {
        this.primBounds = primBounds;
        this.result = new Int32Array(Math.max(16, primCount));
        this._build(primCount, maxLeafSize);
    }

    private _build(primCount: number, maxLeafSize: number): void {
        const primBounds = this.primBounds;
        const prims = new Int32Array(primCount);
        for (let i = 0; i < primCount; i++) prims[i] = i;
        const centres = new Float64Array(primCount * 3);
        for (let i = 0; i < primCount; i++) {
            const b = i * 6;
            centres[i * 3] = (primBounds[b] + primBounds[b + 3]) / 2;
            centres[i * 3 + 1] = (primBounds[b + 1] + primBounds[b + 4]) / 2;
            centres[i * 3 + 2] = (primBounds[b + 2] + primBounds[b + 5]) / 2;
        }

        const capacity = Math.max(1, primCount * 2);
        const bounds = new Float64Array(capacity * 6);
        const offsetOrRight = new Int32Array(capacity);
        const leafCounts = new Int32Array(capacity);
        let nodeCount = 0;
        const maxDepth = 40;

        const splitNode = (
            offset: number,
            length: number,
            depth: number,
        ): number => {
            const node = nodeCount++;
            const end = offset + length;
            let minX = Infinity;
            let minY = Infinity;
            let minZ = Infinity;
            let maxX = -Infinity;
            let maxY = -Infinity;
            let maxZ = -Infinity;
            let cMinX = Infinity;
            let cMinY = Infinity;
            let cMinZ = Infinity;
            let cMaxX = -Infinity;
            let cMaxY = -Infinity;
            let cMaxZ = -Infinity;
            for (let i = offset; i < end; i++) {
                const prim = prims[i];
                const b = prim * 6;
                if (primBounds[b] < minX) minX = primBounds[b];
                if (primBounds[b + 1] < minY) minY = primBounds[b + 1];
                if (primBounds[b + 2] < minZ) minZ = primBounds[b + 2];
                if (primBounds[b + 3] > maxX) maxX = primBounds[b + 3];
                if (primBounds[b + 4] > maxY) maxY = primBounds[b + 4];
                if (primBounds[b + 5] > maxZ) maxZ = primBounds[b + 5];
                const c = prim * 3;
                if (centres[c] < cMinX) cMinX = centres[c];
                if (centres[c] > cMaxX) cMaxX = centres[c];
                if (centres[c + 1] < cMinY) cMinY = centres[c + 1];
                if (centres[c + 1] > cMaxY) cMaxY = centres[c + 1];
                if (centres[c + 2] < cMinZ) cMinZ = centres[c + 2];
                if (centres[c + 2] > cMaxZ) cMaxZ = centres[c + 2];
            }
            const nb = node * 6;
            bounds[nb] = minX;
            bounds[nb + 1] = minY;
            bounds[nb + 2] = minZ;
            bounds[nb + 3] = maxX;
            bounds[nb + 4] = maxY;
            bounds[nb + 5] = maxZ;

            const makeLeaf = (): number => {
                offsetOrRight[node] = offset;
                leafCounts[node] = length;
                return node;
            };
            if (length <= maxLeafSize || depth >= maxDepth) return makeLeaf();

            let axis = 0;
            let extent = cMaxX - cMinX;
            let mid = (cMinX + cMaxX) / 2;
            if (cMaxY - cMinY > extent) {
                axis = 1;
                extent = cMaxY - cMinY;
                mid = (cMinY + cMaxY) / 2;
            }
            if (cMaxZ - cMinZ > extent) {
                axis = 2;
                extent = cMaxZ - cMinZ;
                mid = (cMinZ + cMaxZ) / 2;
            }
            if (!(extent > 0)) return makeLeaf();

            let left = offset;
            let right = end - 1;
            for (;;) {
                while (left <= right && centres[prims[left] * 3 + axis] < mid) {
                    left++;
                }
                while (
                    left <= right &&
                    centres[prims[right] * 3 + axis] >= mid
                ) {
                    right--;
                }
                if (left < right) {
                    const t = prims[left];
                    prims[left] = prims[right];
                    prims[right] = t;
                    left++;
                    right--;
                } else {
                    break;
                }
            }
            const leftLength = left - offset;
            if (leftLength === 0 || leftLength === length) return makeLeaf();

            splitNode(offset, leftLength, depth + 1);
            const rightNode = splitNode(left, length - leftLength, depth + 1);
            offsetOrRight[node] = rightNode;
            leafCounts[node] = 0;
            return node;
        };

        splitNode(0, primCount, 0);
        this.prims = prims;
        this.bounds = bounds;
        this.offsetOrRight = offsetOrRight;
        this.leafCounts = leafCounts;
        this.stack = new Int32Array(maxDepth * 2 + 8);
    }

    /**
     * Collects the prims whose box, grown by `radius`, is reached by the ray
     * for a parameter in [-radius, tMax]. The direction must be normalised.
     * Results are written to `this.result`; returns their count.
     */
    collect(
        ox: number,
        oy: number,
        oz: number,
        dx: number,
        dy: number,
        dz: number,
        radius: number,
        tMax: number,
    ): number {
        const bounds = this.bounds;
        const primBounds = this.primBounds;
        const offsetOrRight = this.offsetOrRight;
        const leafCounts = this.leafCounts;
        const prims = this.prims;
        const stack = this.stack;
        const result = this.result;
        const invX = 1 / dx;
        const invY = 1 / dy;
        const invZ = 1 / dz;
        const tLow = -radius;

        // Slab test of box `b` (index into `box`, grown by radius). A NaN
        // (0 * Infinity for an axis-parallel ray on a slab plane) is ignored
        // so it never narrows the interval.
        const reaches = (box: Float64Array, b: number): boolean => {
            let tmin = tLow;
            let tmax = tMax;
            let t1: number;
            let t2: number;
            if (invX >= 0) {
                t1 = (box[b] - radius - ox) * invX;
                t2 = (box[b + 3] + radius - ox) * invX;
            } else {
                t1 = (box[b + 3] + radius - ox) * invX;
                t2 = (box[b] - radius - ox) * invX;
            }
            if (t1 > tmin) tmin = t1;
            if (t2 < tmax) tmax = t2;
            if (tmin > tmax) return false;
            if (invY >= 0) {
                t1 = (box[b + 1] - radius - oy) * invY;
                t2 = (box[b + 4] + radius - oy) * invY;
            } else {
                t1 = (box[b + 4] + radius - oy) * invY;
                t2 = (box[b + 1] - radius - oy) * invY;
            }
            if (t1 > tmin) tmin = t1;
            if (t2 < tmax) tmax = t2;
            if (tmin > tmax) return false;
            if (invZ >= 0) {
                t1 = (box[b + 2] - radius - oz) * invZ;
                t2 = (box[b + 5] + radius - oz) * invZ;
            } else {
                t1 = (box[b + 5] + radius - oz) * invZ;
                t2 = (box[b + 2] - radius - oz) * invZ;
            }
            if (t1 > tmin) tmin = t1;
            if (t2 < tmax) tmax = t2;
            return tmin <= tmax;
        };

        let count = 0;
        let sp = 0;
        stack[sp++] = 0;

        // The stack only holds tree nodes (at most one pending sibling per
        // level); leaf prims are tested in place.
        while (sp > 0) {
            const node = stack[--sp];
            if (!reaches(bounds, node * 6)) continue;

            const leafCount = leafCounts[node];
            if (leafCount === 0) {
                stack[sp++] = offsetOrRight[node];
                stack[sp++] = node + 1;
                continue;
            }
            const start = offsetOrRight[node];
            for (let i = start, end = start + leafCount; i < end; i++) {
                const prim = prims[i];
                if (reaches(primBounds, prim * 6)) result[count++] = prim;
            }
        }
        return count;
    }
}
