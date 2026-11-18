/**
 * PointsBVH: a bounding volume hierarchy over the points of one geometry.
 *
 * It follows the design of three-mesh-bvh's MeshBVH, adapted from triangles to
 * points:
 *  - Build: split a node on the longest axis of its bounds at the centre
 *    (three-mesh-bvh's CENTER strategy), partitioning an index array in place
 *    (three-mesh-bvh's "indirect" mode), so the geometry itself is never
 *    reordered or modified.
 *  - Storage: flat typed arrays with nodes in depth-first order, so the left
 *    child of node `n` is always `n + 1` and only the right child needs a
 *    pointer (the same packing three-mesh-bvh uses).
 *  - Query: a ray/box slab test that only descends into nodes the ray can
 *    reach. Points have no size, so each box is grown by the pick radius
 *    before the test. Any point within that radius of the ray is then always
 *    inside a box the ray passes through, so the pruning never drops a hit.
 *
 * All coordinates are in the geometry's local space. The point test inside a
 * leaf is the exact arithmetic of three.js Ray.distanceSqToPoint, so a point
 * reported here is a point Points.raycast would accept.
 */

import type { HitIndexBuffer, PointsBVHOptions, PositionArray } from './types';

// 32 points per leaf measured best on a real 1.5M point Potree cloud: half
// the build time of 16 for the same query time.
const DEFAULT_MAX_LEAF_SIZE = 32;
const DEFAULT_MAX_DEPTH = 40;

/**
 * Bounds of the points with finite coordinates, as
 * [minX, minY, minZ, maxX, maxY, maxZ], or null when there is none.
 * Points with a NaN/Infinity coordinate never pass a distance or box test,
 * so they are left out.
 *
 * @param positions xyz triplets
 * @param count number of triplets to read
 */
export function computeFinitePointBounds(
    positions: PositionArray,
    count: number,
): number[] | null {
    let minX = Infinity;
    let minY = Infinity;
    let minZ = Infinity;
    let maxX = -Infinity;
    let maxY = -Infinity;
    let maxZ = -Infinity;
    let finite = 0;
    for (let i = 0, p = 0; i < count; i++, p += 3) {
        const x = positions[p];
        const y = positions[p + 1];
        const z = positions[p + 2];
        if (x - x !== 0 || y - y !== 0 || z - z !== 0) continue;
        finite++;
        if (x < minX) minX = x;
        if (x > maxX) maxX = x;
        if (y < minY) minY = y;
        if (y > maxY) maxY = y;
        if (z < minZ) minZ = z;
        if (z > maxZ) maxZ = z;
    }
    return finite ? [minX, minY, minZ, maxX, maxY, maxZ] : null;
}

// Shared traversal stack. Queries never nest, so one stack is enough.
// Traversal pushes at most one entry per tree level.
let traversalStack = new Int32Array(DEFAULT_MAX_DEPTH * 2 + 8);

export class PointsBVH {
    positions: PositionArray;
    count: number;
    maxLeafSize: number;
    maxDepth: number;
    indices!: Uint32Array;
    nodeCount!: number;
    bounds!: Float64Array;
    offsetOrRight!: Uint32Array;
    leafCounts!: Uint32Array;

    /**
     * @param positions xyz triplets (e.g. position.array)
     * @param count number of points (position.count)
     */
    constructor(
        positions: PositionArray,
        count: number,
        options: PointsBVHOptions = {},
    ) {
        this.positions = positions;
        this.count = count;
        this.maxLeafSize = options.maxLeafSize ?? DEFAULT_MAX_LEAF_SIZE;
        this.maxDepth = options.maxDepth ?? DEFAULT_MAX_DEPTH;
        if (traversalStack.length < this.maxDepth * 2 + 8) {
            traversalStack = new Int32Array(this.maxDepth * 2 + 8);
        }
        this._build();
    }

    private _build(): void {
        const positions = this.positions;
        const count = this.count;

        // Build-time copies of the coordinates, permuted together with
        // `indices`, so every pass reads contiguous memory (three-mesh-bvh
        // permutes its triangle bounds the same way). Float32 data is copied
        // into Float32Arrays, so the values are exactly the source values.
        const CoordArray =
            positions instanceof Float32Array ? Float32Array : Float64Array;
        const indices = new Uint32Array(count);
        const xs = new CoordArray(count);
        const ys = new CoordArray(count);
        const zs = new CoordArray(count);

        // Points with NaN or Infinity coordinates can never pass the
        // distance test (the comparison is false), so they are left out.
        let finiteCount = 0;
        for (let i = 0, p = 0; i < count; i++, p += 3) {
            const x = positions[p];
            const y = positions[p + 1];
            const z = positions[p + 2];
            if (x - x === 0 && y - y === 0 && z - z === 0) {
                indices[finiteCount] = i;
                xs[finiteCount] = x;
                ys[finiteCount] = y;
                zs[finiteCount] = z;
                finiteCount++;
            }
        }

        let capacity = Math.max(
            8,
            Math.ceil((finiteCount / this.maxLeafSize) * 4),
        );
        let bounds = new Float64Array(capacity * 6);
        // Leaf: start offset into `indices`. Inner node: right child index.
        let offsetOrRight = new Uint32Array(capacity);
        // Leaf: number of points (always > 0). Inner node: 0.
        let leafCounts = new Uint32Array(capacity);
        let nodeCount = 0;

        const maxLeafSize = this.maxLeafSize;
        const maxDepth = this.maxDepth;

        const allocNode = (): number => {
            if (nodeCount === capacity) {
                capacity *= 2;
                const nb = new Float64Array(capacity * 6);
                nb.set(bounds);
                bounds = nb;
                const no = new Uint32Array(capacity);
                no.set(offsetOrRight);
                offsetOrRight = no;
                const nc = new Uint32Array(capacity);
                nc.set(leafCounts);
                leafCounts = nc;
            }
            return nodeCount++;
        };

        const splitNode = (
            offset: number,
            length: number,
            depth: number,
        ): number => {
            const node = allocNode();
            const end = offset + length;

            let minX = Infinity;
            let minY = Infinity;
            let minZ = Infinity;
            let maxX = -Infinity;
            let maxY = -Infinity;
            let maxZ = -Infinity;
            for (let i = offset; i < end; i++) {
                const x = xs[i];
                const y = ys[i];
                const z = zs[i];
                if (x < minX) minX = x;
                if (x > maxX) maxX = x;
                if (y < minY) minY = y;
                if (y > maxY) maxY = y;
                if (z < minZ) minZ = z;
                if (z > maxZ) maxZ = z;
            }
            const b = node * 6;
            bounds[b] = minX;
            bounds[b + 1] = minY;
            bounds[b + 2] = minZ;
            bounds[b + 3] = maxX;
            bounds[b + 4] = maxY;
            bounds[b + 5] = maxZ;

            if (length <= maxLeafSize || depth >= maxDepth) {
                offsetOrRight[node] = offset;
                leafCounts[node] = length;
                return node;
            }

            const ex = maxX - minX;
            const ey = maxY - minY;
            const ez = maxZ - minZ;
            let axis = 0;
            let extent = ex;
            let mid = (minX + maxX) / 2;
            if (ey > extent) {
                axis = 1;
                extent = ey;
                mid = (minY + maxY) / 2;
            }
            if (ez > extent) {
                axis = 2;
                extent = ez;
                mid = (minZ + maxZ) / 2;
            }

            // Hoare partition: coordinate < mid goes left, the rest right.
            const axisCoords = axis === 0 ? xs : axis === 1 ? ys : zs;
            let left = offset;
            let right = end - 1;
            if (extent > 0) {
                for (;;) {
                    while (left <= right && axisCoords[left] < mid) left++;
                    while (left <= right && axisCoords[right] >= mid) right--;
                    if (left < right) {
                        const ti = indices[left];
                        indices[left] = indices[right];
                        indices[right] = ti;
                        const tx = xs[left];
                        xs[left] = xs[right];
                        xs[right] = tx;
                        const ty = ys[left];
                        ys[left] = ys[right];
                        ys[right] = ty;
                        const tz = zs[left];
                        zs[left] = zs[right];
                        zs[right] = tz;
                        left++;
                        right--;
                    } else {
                        break;
                    }
                }
            }

            const leftLength = left - offset;
            if (!(extent > 0) || leftLength === 0 || leftLength === length) {
                // All points coincide on every axis, or the split is degenerate.
                offsetOrRight[node] = offset;
                leafCounts[node] = length;
                return node;
            }

            // Depth-first layout: the left subtree is allocated right after
            // this node, so it starts at node + 1.
            splitNode(offset, leftLength, depth + 1);
            // Store the result in a local first: building the right subtree
            // can grow (reallocate) the node arrays, and
            // `offsetOrRight[node] = splitNode(...)` would resolve the array
            // before the call and write into the discarded one.
            const rightNode = splitNode(left, length - leftLength, depth + 1);
            offsetOrRight[node] = rightNode;
            leafCounts[node] = 0;
            return node;
        };

        if (finiteCount > 0) splitNode(0, finiteCount, 0);

        // Only the permuted indices are kept; queries read the source array.
        this.indices =
            finiteCount === count ? indices : indices.slice(0, finiteCount);
        this.nodeCount = nodeCount;
        this.bounds = bounds.slice(0, nodeCount * 6);
        this.offsetOrRight = offsetOrRight.slice(0, nodeCount);
        this.leafCounts = leafCounts.slice(0, nodeCount);
    }

    /**
     * Collects the indices of points whose squared distance to the ray is below
     * `thresholdSq`, using the exact arithmetic of three.js
     * Ray.distanceSqToPoint (identical in three r124 and r161).
     *
     * `radius` (>= sqrt(thresholdSq)) and `tMax` only prune the traversal, so
     * they must err on the larger side. A point behind the ray origin is
     * measured to the origin itself, as three.js does, so the ray is treated
     * as reaching back to t = -radius.
     *
     * @param ox ray origin (local space)
     * @param dx normalised ray direction (local space)
     * @param radius conservative pruning radius
     * @param tMax conservative maximum ray parameter (Infinity allowed)
     * @param thresholdSq exact squared distance threshold
     * @param out receives indices (grown when needed)
     * @returns number of indices written to out.array
     */
    collectRayHits(
        ox: number,
        oy: number,
        oz: number,
        dx: number,
        dy: number,
        dz: number,
        radius: number,
        tMax: number,
        thresholdSq: number,
        out: HitIndexBuffer,
    ): number {
        if (!this.nodeCount) return 0;

        const bounds = this.bounds;
        const offsetOrRight = this.offsetOrRight;
        const leafCounts = this.leafCounts;
        const indices = this.indices;
        const positions = this.positions;
        const stack = traversalStack;

        const invX = 1 / dx;
        const invY = 1 / dy;
        const invZ = 1 / dz;
        const tLow = -radius;

        let hitCount = 0;
        let outArray = out.array;
        let sp = 0;
        stack[sp++] = 0;

        while (sp > 0) {
            const node = stack[--sp];
            const b = node * 6;

            // Slab test against the box grown by `radius`. Comparisons are
            // written so a NaN (0 * Infinity for an axis-parallel ray lying on
            // a slab plane) never narrows the interval.
            let tmin = tLow;
            let tmax = tMax;
            let t1: number;
            let t2: number;

            if (invX >= 0) {
                t1 = (bounds[b] - radius - ox) * invX;
                t2 = (bounds[b + 3] + radius - ox) * invX;
            } else {
                t1 = (bounds[b + 3] + radius - ox) * invX;
                t2 = (bounds[b] - radius - ox) * invX;
            }
            if (t1 > tmin) tmin = t1;
            if (t2 < tmax) tmax = t2;
            if (tmin > tmax) continue;

            if (invY >= 0) {
                t1 = (bounds[b + 1] - radius - oy) * invY;
                t2 = (bounds[b + 4] + radius - oy) * invY;
            } else {
                t1 = (bounds[b + 4] + radius - oy) * invY;
                t2 = (bounds[b + 1] - radius - oy) * invY;
            }
            if (t1 > tmin) tmin = t1;
            if (t2 < tmax) tmax = t2;
            if (tmin > tmax) continue;

            if (invZ >= 0) {
                t1 = (bounds[b + 2] - radius - oz) * invZ;
                t2 = (bounds[b + 5] + radius - oz) * invZ;
            } else {
                t1 = (bounds[b + 5] + radius - oz) * invZ;
                t2 = (bounds[b + 2] - radius - oz) * invZ;
            }
            if (t1 > tmin) tmin = t1;
            if (t2 < tmax) tmax = t2;
            if (tmin > tmax) continue;

            const leafCount = leafCounts[node];
            if (leafCount === 0) {
                stack[sp++] = offsetOrRight[node];
                stack[sp++] = node + 1;
                continue;
            }

            const start = offsetOrRight[node];
            const end = start + leafCount;
            for (let i = start; i < end; i++) {
                const index = indices[i];
                const p = 3 * index;
                const px = positions[p];
                const py = positions[p + 1];
                const pz = positions[p + 2];

                // three.js Ray.distanceSqToPoint, same operation order.
                const directionDistance =
                    (px - ox) * dx + (py - oy) * dy + (pz - oz) * dz;
                let distanceSq: number;
                if (directionDistance < 0) {
                    const ex = ox - px;
                    const ey = oy - py;
                    const ez = oz - pz;
                    distanceSq = ex * ex + ey * ey + ez * ez;
                } else {
                    const fx = dx * directionDistance + ox - px;
                    const fy = dy * directionDistance + oy - py;
                    const fz = dz * directionDistance + oz - pz;
                    distanceSq = fx * fx + fy * fy + fz * fz;
                }

                if (distanceSq < thresholdSq) {
                    if (hitCount === outArray.length) {
                        const grown = new Uint32Array(outArray.length * 2);
                        grown.set(outArray);
                        outArray = grown;
                        out.array = grown;
                    }
                    outArray[hitCount++] = index;
                }
            }
        }

        return hitCount;
    }
}
