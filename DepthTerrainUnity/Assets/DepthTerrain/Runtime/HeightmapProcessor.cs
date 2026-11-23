using System;
using System.Threading.Tasks;
using UnityEngine;

namespace DepthTerrain
{
    /// <summary>
    /// High-performance pure C# 16-bit heightmap decoding, resampling, ping-pong Gaussian filtering,
    /// slope-aware cliff relief, and multi-core DXT5nm normal-map generation for Unity TerrainData.
    /// </summary>
    public static class HeightmapProcessor
    {
        private static readonly int[] ValidResolutions = { 33, 65, 129, 257, 513, 1025, 2049 };

        /// <summary>
        /// Minimum grid dimension at which multi-core Parallel.For dispatch outweighs thread-pool overhead.
        /// </summary>
        public const int ParallelThreshold = 129;

        // Main-thread reusable pixel buffer to avoid multi-MB GC churn during live slider drags.
        private static Color32[] _reusableNormalPixels;

        /// <summary>
        /// Snap any requested heightmap dimension to the closest valid Unity TerrainData resolution (2^n + 1).
        /// </summary>
        public static int NearestUnityResolution(int requested)
        {
            int best = ValidResolutions[0];
            int bestDiff = Math.Abs(requested - best);
            for (int i = 1; i < ValidResolutions.Length; i++)
            {
                int diff = Math.Abs(requested - ValidResolutions[i]);
                if (diff < bestDiff)
                {
                    bestDiff = diff;
                    best = ValidResolutions[i];
                }
            }
            return best;
        }

        /// <summary>
        /// Decode 16-bit RAW heightmap bytes into Unity's [z, x] float array in [0, 1].
        /// When <paramref name="flipY"/> is true, top image row (North, y=0) maps to z = resolution - 1 (+Z),
        /// matching Unity's TerrainLayer texture UV orientation pixel-for-pixel.
        /// </summary>
        public static float[,] DecodeRaw16(
            byte[] rawBytes,
            int resolution,
            bool littleEndian = true,
            bool flipY = true,
            bool useParallel = true
        )
        {
            if (rawBytes == null)
                throw new ArgumentNullException(nameof(rawBytes));
            if (resolution <= 1)
                throw new ArgumentOutOfRangeException(nameof(resolution));

            int expectedBytes = resolution * resolution * 2;
            if (rawBytes.Length < expectedBytes)
            {
                throw new ArgumentException(
                    $"RAW heightmap buffer too small: expected at least {expectedBytes} bytes for {resolution}x{resolution}x16-bit, got {rawBytes.Length}."
                );
            }

            var heights = new float[resolution, resolution];
            const float invMax = 1f / 65535f;

            if (useParallel && resolution >= ParallelThreshold)
            {
                Parallel.For(0, resolution, row => DecodeRaw16Row(rawBytes, heights, row, resolution, littleEndian, flipY, invMax));
            }
            else
            {
                for (int row = 0; row < resolution; row++)
                {
                    DecodeRaw16Row(rawBytes, heights, row, resolution, littleEndian, flipY, invMax);
                }
            }

            return heights;
        }

        private static void DecodeRaw16Row(
            byte[] rawBytes,
            float[,] heights,
            int row,
            int resolution,
            bool littleEndian,
            bool flipY,
            float invMax
        )
        {
            int dstZ = flipY ? (resolution - 1 - row) : row;
            int rowOffset = row * resolution * 2;
            if (littleEndian)
            {
                for (int col = 0; col < resolution; col++)
                {
                    int idx = rowOffset + (col << 1);
                    ushort sample = (ushort)(rawBytes[idx] | (rawBytes[idx + 1] << 8));
                    heights[dstZ, col] = sample * invMax;
                }
            }
            else
            {
                for (int col = 0; col < resolution; col++)
                {
                    int idx = rowOffset + (col << 1);
                    ushort sample = (ushort)((rawBytes[idx] << 8) | rawBytes[idx + 1]);
                    heights[dstZ, col] = sample * invMax;
                }
            }
        }

        /// <summary>
        /// Encode a Unity [z, x] float heightmap back to 16-bit little-endian RAW bytes (top-to-bottom).
        /// </summary>
        public static byte[] EncodeRaw16(float[,] heights, bool flipY = true)
        {
            int res = heights.GetLength(0);
            var bytes = new byte[res * res * 2];
            for (int row = 0; row < res; row++)
            {
                int srcZ = flipY ? (res - 1 - row) : row;
                int rowOffset = row * res * 2;
                for (int col = 0; col < res; col++)
                {
                    float clamped = Clamp01Fast(heights[srcZ, col]);
                    ushort val = (ushort)Math.Round(clamped * 65535f);
                    int idx = rowOffset + (col << 1);
                    bytes[idx] = (byte)(val & 0xFF);
                    bytes[idx + 1] = (byte)((val >> 8) & 0xFF);
                }
            }
            return bytes;
        }

        /// <summary>
        /// Apply all non-destructive terrain shaping transforms (resample, invert, ping-pong smooth,
        /// optional slope-aware cliff detail, gamma, floor clamp, terracing, optional edge falloff)
        /// without mutating the source array.
        /// </summary>
        public static float[,] ProcessHeights(
            float[,] sourceHeights,
            TerrainBuildSettings settings,
            bool useParallel = true
        )
        {
            if (sourceHeights == null)
                throw new ArgumentNullException(nameof(sourceHeights));
            if (settings == null)
                throw new ArgumentNullException(nameof(settings));

            int srcRes = sourceHeights.GetLength(0);
            int targetRes = NearestUnityResolution(settings.targetResolution);
            bool runParallel = useParallel && targetRes >= ParallelThreshold;

            float[,] working;
            if (srcRes == targetRes)
            {
                working = new float[targetRes, targetRes];
                if (settings.invertHeight)
                {
                    if (runParallel)
                    {
                        Parallel.For(0, targetRes, z =>
                        {
                            for (int x = 0; x < targetRes; x++)
                                working[z, x] = 1f - sourceHeights[z, x];
                        });
                    }
                    else
                    {
                        for (int z = 0; z < targetRes; z++)
                        for (int x = 0; x < targetRes; x++)
                            working[z, x] = 1f - sourceHeights[z, x];
                    }
                }
                else
                {
                    Array.Copy(sourceHeights, working, targetRes * targetRes);
                }
            }
            else
            {
                working = ResampleBilinear(sourceHeights, targetRes, useParallel);
                if (settings.invertHeight)
                {
                    if (runParallel)
                    {
                        Parallel.For(0, targetRes, z =>
                        {
                            for (int x = 0; x < targetRes; x++)
                                working[z, x] = 1f - working[z, x];
                        });
                    }
                    else
                    {
                        for (int z = 0; z < targetRes; z++)
                        for (int x = 0; x < targetRes; x++)
                            working[z, x] = 1f - working[z, x];
                    }
                }
            }

            int n = working.GetLength(0);
            float[,] scratch = null;

            if (settings.smoothingPasses > 0)
            {
                scratch = new float[n, n];
                GaussianSmoothInPlace(working, settings.smoothingPasses, scratch, useParallel);
            }

            if (settings.cliffDetailStrength > 0.001f)
            {
                scratch ??= new float[n, n];
                ApplyCliffDetailReliefInPlace(working, settings.cliffDetailStrength, scratch, useParallel);
            }

            float gamma = ClampFast(settings.elevationGamma, 0.1f, 5f);
            bool applyGamma = Math.Abs(gamma - 1f) > 1e-4f;
            float clampFloor = ClampFast(settings.floorClamp, 0f, 0.9f);
            bool applyFloor = clampFloor > 0f;
            float floorScale = 1f / Math.Max(1f - clampFloor, 1e-4f);
            int steps = Math.Max(0, settings.terraceSteps);
            bool applyTerrace = steps > 1;
            float invStepsMinusOne = applyTerrace ? (1f / (steps - 1)) : 0f;
            bool doFalloff = settings.edgeFalloff;
            float margin = ClampFast(settings.edgeFalloffMargin, 0.01f, 0.45f);
            float invN = 1f / Math.Max(n - 1, 1);

            float[] edgeMask = null;
            if (doFalloff)
            {
                edgeMask = new float[n];
                for (int i = 0; i < n; i++)
                    edgeMask[i] = ComputeEdgeMask(i * invN, margin);
            }

            if (runParallel)
            {
                Parallel.For(0, n, z =>
                {
                    ShapeRow(
                        working,
                        z,
                        n,
                        applyGamma,
                        gamma,
                        applyFloor,
                        clampFloor,
                        floorScale,
                        applyTerrace,
                        steps,
                        invStepsMinusOne,
                        doFalloff,
                        edgeMask
                    );
                });
            }
            else
            {
                for (int z = 0; z < n; z++)
                {
                    ShapeRow(
                        working,
                        z,
                        n,
                        applyGamma,
                        gamma,
                        applyFloor,
                        clampFloor,
                        floorScale,
                        applyTerrace,
                        steps,
                        invStepsMinusOne,
                        doFalloff,
                        edgeMask
                    );
                }
            }

            return working;
        }

        private static void ShapeRow(
            float[,] working,
            int z,
            int n,
            bool applyGamma,
            float gamma,
            bool applyFloor,
            float clampFloor,
            float floorScale,
            bool applyTerrace,
            int steps,
            float invStepsMinusOne,
            bool doFalloff,
            float[] edgeMask
        )
        {
            float fz = doFalloff ? edgeMask[z] : 1f;

            for (int x = 0; x < n; x++)
            {
                float h = Clamp01Fast(working[z, x]);

                if (applyGamma)
                    h = h <= 0f ? 0f : (float)Math.Pow(h, gamma);

                if (applyFloor)
                    h = h <= clampFloor ? 0f : (h - clampFloor) * floorScale;

                if (applyTerrace)
                {
                    float stepped = Clamp01Fast((float)Math.Floor(h * steps) * invStepsMinusOne);
                    h = h + (stepped - h) * 0.7f;
                }

                if (doFalloff)
                {
                    h *= edgeMask[x] * fz;
                }

                working[z, x] = Clamp01Fast(h);
            }
        }

        private static float ComputeEdgeMask(float t, float margin)
        {
            float d = Math.Min(t, 1f - t);
            if (d >= margin) return 1f;
            float u = Clamp01Fast(d / margin);
            return u * u * (3f - 2f * u);
        }

        /// <summary>
        /// Bilinearly resample a square heightmap to a new square resolution.
        /// </summary>
        public static float[,] ResampleBilinear(float[,] src, int dstRes, bool useParallel = true)
        {
            int srcRes = src.GetLength(0);
            var dst = new float[dstRes, dstRes];
            float scale = (srcRes - 1f) / Math.Max(dstRes - 1f, 1f);
            int maxSrc = srcRes - 1;

            if (useParallel && dstRes >= ParallelThreshold)
            {
                Parallel.For(0, dstRes, z => ResampleBilinearRow(src, dst, z, dstRes, maxSrc, scale));
            }
            else
            {
                for (int z = 0; z < dstRes; z++)
                {
                    ResampleBilinearRow(src, dst, z, dstRes, maxSrc, scale);
                }
            }

            return dst;
        }

        private static void ResampleBilinearRow(float[,] src, float[,] dst, int z, int dstRes, int maxSrc, float scale)
        {
            float sz = z * scale;
            int z0 = (int)sz;
            if (z0 < 0) z0 = 0;
            else if (z0 > maxSrc) z0 = maxSrc;
            int z1 = z0 < maxSrc ? z0 + 1 : maxSrc;
            float tz = sz - z0;

            for (int x = 0; x < dstRes; x++)
            {
                float sx = x * scale;
                int x0 = (int)sx;
                if (x0 < 0) x0 = 0;
                else if (x0 > maxSrc) x0 = maxSrc;
                int x1 = x0 < maxSrc ? x0 + 1 : maxSrc;
                float tx = sx - x0;

                float h00 = src[z0, x0];
                float h10 = src[z0, x1];
                float h01 = src[z1, x0];
                float h11 = src[z1, x1];

                float h0 = h00 + (h10 - h00) * tx;
                float h1 = h01 + (h11 - h01) * tx;
                dst[z, x] = h0 + (h1 - h0) * tz;
            }
        }

        /// <summary>
        /// Single separable [1, 2, 1] / 4 Gaussian blur pass returning a new array.
        /// For multi-pass smoothing with zero per-pass allocations, prefer <see cref="GaussianSmoothInPlace"/>.
        /// </summary>
        public static float[,] GaussianSmoothPass(float[,] src, bool useParallel = true)
        {
            int n = src.GetLength(0);
            var dst = new float[n, n];
            Array.Copy(src, dst, n * n);
            GaussianSmoothInPlace(dst, passes: 1, scratch: null, useParallel: useParallel);
            return dst;
        }

        /// <summary>
        /// Multi-pass separable [1, 2, 1] / 4 Gaussian blur operating in-place via a single ping-pong
        /// scratch buffer (horizontal: data -&gt; scratch, vertical: scratch -&gt; data) with row-major
        /// cache-friendly traversal and multi-core Parallel.For acceleration.
        /// </summary>
        public static void GaussianSmoothInPlace(
            float[,] data,
            int passes,
            float[,] scratch = null,
            bool useParallel = true
        )
        {
            if (data == null)
                throw new ArgumentNullException(nameof(data));
            if (passes <= 0)
                return;

            int n = data.GetLength(0);
            if (n < 2)
                return;

            float[,] tmp = (scratch != null && scratch.GetLength(0) == n && scratch.GetLength(1) == n)
                ? scratch
                : new float[n, n];

            bool runParallel = useParallel && n >= ParallelThreshold;
            int last = n - 1;

            for (int p = 0; p < passes; p++)
            {
                // Horizontal pass: data -> tmp (row-major, zero aliasing)
                if (runParallel)
                {
                    Parallel.For(0, n, z => SmoothHorizontalRow(data, tmp, z, last));
                }
                else
                {
                    for (int z = 0; z < n; z++)
                        SmoothHorizontalRow(data, tmp, z, last);
                }

                // Vertical pass: tmp -> data (row-major z outer / x inner for L1 cache streaming)
                if (runParallel)
                {
                    Parallel.For(0, n, z => SmoothVerticalRow(tmp, data, z, n, last));
                }
                else
                {
                    for (int z = 0; z < n; z++)
                        SmoothVerticalRow(tmp, data, z, n, last);
                }
            }
        }

        private static void SmoothHorizontalRow(float[,] src, float[,] dst, int z, int last)
        {
            dst[z, 0] = (src[z, 0] * 3f + src[z, 1]) * 0.25f;
            for (int x = 1; x < last; x++)
            {
                dst[z, x] = (src[z, x - 1] + src[z, x] * 2f + src[z, x + 1]) * 0.25f;
            }
            dst[z, last] = (src[z, last - 1] + src[z, last] * 3f) * 0.25f;
        }

        private static void SmoothVerticalRow(float[,] src, float[,] dst, int z, int n, int last)
        {
            if (z == 0)
            {
                for (int x = 0; x < n; x++)
                    dst[0, x] = (src[0, x] * 3f + src[1, x]) * 0.25f;
            }
            else if (z == last)
            {
                for (int x = 0; x < n; x++)
                    dst[last, x] = (src[last - 1, x] + src[last, x] * 3f) * 0.25f;
            }
            else
            {
                int zm = z - 1;
                int zp = z + 1;
                for (int x = 0; x < n; x++)
                    dst[z, x] = (src[zm, x] + src[z, x] * 2f + src[zp, x]) * 0.25f;
            }
        }

        /// <summary>
        /// Slope-aware local ridge/cliff sharpening that accentuates steep escarpments and crater rims
        /// while preserving smooth plains and coastlines without altering satellite albedo mapping.
        /// </summary>
        public static void ApplyCliffDetailReliefInPlace(
            float[,] data,
            float strength,
            float[,] scratch = null,
            bool useParallel = true
        )
        {
            if (data == null || strength <= 0.001f)
                return;

            int n = data.GetLength(0);
            if (n < 3)
                return;

            float[,] copy = (scratch != null && scratch.GetLength(0) == n && scratch.GetLength(1) == n)
                ? scratch
                : new float[n, n];
            Array.Copy(data, copy, n * n);

            bool runParallel = useParallel && n >= ParallelThreshold;
            int last = n - 1;
            float slopeScale = 0.5f * n;
            float clampedStrength = ClampFast(strength, 0f, 2f);

            if (runParallel)
            {
                Parallel.For(0, n, z => CliffDetailRow(copy, data, z, n, last, slopeScale, clampedStrength));
            }
            else
            {
                for (int z = 0; z < n; z++)
                    CliffDetailRow(copy, data, z, n, last, slopeScale, clampedStrength);
            }
        }

        private static void CliffDetailRow(
            float[,] src,
            float[,] dst,
            int z,
            int n,
            int last,
            float slopeScale,
            float strength
        )
        {
            int zm = z > 0 ? z - 1 : 0;
            int zp = z < last ? z + 1 : last;

            for (int x = 0; x < n; x++)
            {
                int xm = x > 0 ? x - 1 : 0;
                int xp = x < last ? x + 1 : last;

                float h = src[z, x];
                float hL = src[z, xm];
                float hR = src[z, xp];
                float hD = src[zm, x];
                float hU = src[zp, x];

                float mean = (hL + hR + hD + hU) * 0.25f;
                float detail = h - mean;

                float dhdx = (hR - hL) * slopeScale;
                float dhdz = (hU - hD) * slopeScale;
                float slopeMag = (float)Math.Sqrt(dhdx * dhdx + dhdz * dhdz);
                float slopeWeight = Clamp01Fast(slopeMag * 0.25f);

                // Subtle boost on ridges, stronger contrast on steep cliff faces
                float sharpened = h + detail * (strength * (0.35f + 1.65f * slopeWeight));
                dst[z, x] = Clamp01Fast(sharpened);
            }
        }

        /// <summary>
        /// Bake a tangent-space normal map Texture2D from the processed [z, x] heightmap
        /// to add crisp micro-relief shading to URP Terrain Lit layers.
        /// Set <paramref name="packDxt5nm"/> to true when binding directly to Unity runtime URP shaders on desktop/Metal.
        /// Optionally pass <paramref name="reuseTexture"/> to update an existing RGBA32 texture in-place with zero GPU leak.
        /// </summary>
        public static Texture2D GenerateNormalMap(
            float[,] heights,
            float terrainSize,
            float terrainHeight,
            float strength = 1.5f,
            bool packDxt5nm = false,
            Texture2D reuseTexture = null,
            bool useParallel = true
        )
        {
            int n = heights.GetLength(0);
            Texture2D tex;
            if (reuseTexture != null &&
                reuseTexture.width == n &&
                reuseTexture.height == n &&
                reuseTexture.format == TextureFormat.RGBA32)
            {
                tex = reuseTexture;
            }
            else
            {
                tex = new Texture2D(n, n, TextureFormat.RGBA32, mipChain: true, linear: true)
                {
                    name = "DepthTerrain_GeneratedNormalMap",
                    wrapMode = TextureWrapMode.Clamp,
                    filterMode = FilterMode.Bilinear
                };
            }

            int pixelCount = n * n;
            Color32[] pixels = _reusableNormalPixels;
            if (pixels == null || pixels.Length != pixelCount)
            {
                pixels = new Color32[pixelCount];
                _reusableNormalPixels = pixels;
            }

            float cellSize = terrainSize / Math.Max(n - 1, 1);
            float scale = (terrainHeight / Math.Max(cellSize, 1e-4f)) * Math.Max(strength, 0f);
            float halfScale = 0.5f * scale;
            int last = n - 1;

            if (useParallel && n >= ParallelThreshold)
            {
                Parallel.For(0, n, z => FillNormalMapRow(heights, pixels, z, n, last, halfScale, packDxt5nm));
            }
            else
            {
                for (int z = 0; z < n; z++)
                    FillNormalMapRow(heights, pixels, z, n, last, halfScale, packDxt5nm);
            }

            tex.SetPixels32(pixels);
            tex.Apply(updateMipmaps: true, makeNoLongerReadable: false);
            return tex;
        }

        private static void FillNormalMapRow(
            float[,] heights,
            Color32[] pixels,
            int z,
            int n,
            int last,
            float halfScale,
            bool packDxt5nm
        )
        {
            int zm = z > 0 ? z - 1 : 0;
            int zp = z < last ? z + 1 : last;
            int rowOff = z * n;

            for (int x = 0; x < n; x++)
            {
                int xm = x > 0 ? x - 1 : 0;
                int xp = x < last ? x + 1 : last;

                float dhdx = (heights[z, xp] - heights[z, xm]) * halfScale;
                float dhdz = (heights[zp, x] - heights[zm, x]) * halfScale;

                float nxF = -dhdx;
                float nyF = -dhdz;
                float invLen = 1f / (float)Math.Sqrt(nxF * nxF + nyF * nyF + 1f);
                nxF *= invLen;
                nyF *= invLen;
                float nzF = invLen;

                byte nx = (byte)Math.Round((nxF * 0.5f + 0.5f) * 255f);
                byte ny = (byte)Math.Round((nyF * 0.5f + 0.5f) * 255f);
                byte nz = (byte)Math.Round((nzF * 0.5f + 0.5f) * 255f);

                pixels[rowOff + x] = packDxt5nm
                    ? new Color32(255, ny, 255, nx)
                    : new Color32(nx, ny, nz, 255);
            }
        }

        private static float Clamp01Fast(float v)
        {
            return v < 0f ? 0f : (v > 1f ? 1f : v);
        }

        private static float ClampFast(float v, float min, float max)
        {
            return v < min ? min : (v > max ? max : v);
        }
    }
}
