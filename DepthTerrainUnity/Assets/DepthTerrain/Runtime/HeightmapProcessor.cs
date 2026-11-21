using System;
using UnityEngine;

namespace DepthTerrain
{
    /// <summary>
    /// Pure C# 16-bit heightmap decoding, resampling, filtering, and normal-map generation
    /// for Unity TerrainData.
    /// </summary>
    public static class HeightmapProcessor
    {
        private static readonly int[] ValidResolutions = { 33, 65, 129, 257, 513, 1025, 2049 };

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
        public static float[,] DecodeRaw16(byte[] rawBytes, int resolution, bool littleEndian = true, bool flipY = true)
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

            for (int row = 0; row < resolution; row++)
            {
                int dstZ = flipY ? (resolution - 1 - row) : row;
                int rowOffset = row * resolution * 2;
                for (int col = 0; col < resolution; col++)
                {
                    int idx = rowOffset + col * 2;
                    ushort sample = littleEndian
                        ? (ushort)(rawBytes[idx] | (rawBytes[idx + 1] << 8))
                        : (ushort)((rawBytes[idx] << 8) | rawBytes[idx + 1]);
                    heights[dstZ, col] = sample * invMax;
                }
            }

            return heights;
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
                    ushort val = (ushort)Mathf.RoundToInt(Mathf.Clamp01(heights[srcZ, col]) * 65535f);
                    int idx = rowOffset + col * 2;
                    bytes[idx] = (byte)(val & 0xFF);
                    bytes[idx + 1] = (byte)((val >> 8) & 0xFF);
                }
            }
            return bytes;
        }

        /// <summary>
        /// Apply all non-destructive terrain shaping transforms (resample, invert, smooth, gamma,
        /// floor clamp, terracing, optional edge falloff) without mutating the source array.
        /// </summary>
        public static float[,] ProcessHeights(float[,] sourceHeights, TerrainBuildSettings settings)
        {
            if (sourceHeights == null)
                throw new ArgumentNullException(nameof(sourceHeights));
            if (settings == null)
                throw new ArgumentNullException(nameof(settings));

            int srcRes = sourceHeights.GetLength(0);
            int targetRes = NearestUnityResolution(settings.targetResolution);

            float[,] working = srcRes == targetRes
                ? (float[,])sourceHeights.Clone()
                : ResampleBilinear(sourceHeights, targetRes);

            int n = working.GetLength(0);

            if (settings.invertHeight)
            {
                for (int z = 0; z < n; z++)
                for (int x = 0; x < n; x++)
                    working[z, x] = 1f - working[z, x];
            }

            for (int p = 0; p < settings.smoothingPasses; p++)
            {
                working = GaussianSmoothPass(working);
            }

            float gamma = Mathf.Clamp(settings.elevationGamma, 0.1f, 5f);
            float clampFloor = Mathf.Clamp(settings.floorClamp, 0f, 0.9f);
            float floorScale = 1f / Mathf.Max(1f - clampFloor, 1e-4f);
            int steps = Mathf.Max(0, settings.terraceSteps);
            bool doFalloff = settings.edgeFalloff;
            float margin = Mathf.Clamp(settings.edgeFalloffMargin, 0.01f, 0.45f);
            float invN = 1f / Mathf.Max(n - 1, 1);

            for (int z = 0; z < n; z++)
            {
                float vz = z * invN;
                float fz = doFalloff ? ComputeEdgeMask(vz, margin) : 1f;

                for (int x = 0; x < n; x++)
                {
                    float h = Mathf.Clamp01(working[z, x]);

                    if (Mathf.Abs(gamma - 1f) > 1e-4f)
                        h = Mathf.Pow(h, gamma);

                    if (clampFloor > 0f)
                        h = h <= clampFloor ? 0f : (h - clampFloor) * floorScale;

                    if (steps > 1)
                    {
                        float stepped = Mathf.Floor(h * steps) / (steps - 1);
                        h = Mathf.Lerp(h, Mathf.Clamp01(stepped), 0.7f);
                    }

                    if (doFalloff)
                    {
                        float vx = x * invN;
                        float fx = ComputeEdgeMask(vx, margin);
                        h *= fx * fz;
                    }

                    working[z, x] = Mathf.Clamp01(h);
                }
            }

            return working;
        }

        private static float ComputeEdgeMask(float t, float margin)
        {
            float d = Mathf.Min(t, 1f - t);
            if (d >= margin) return 1f;
            float u = Mathf.Clamp01(d / margin);
            return u * u * (3f - 2f * u);
        }

        /// <summary>
        /// Bilinearly resample a square heightmap to a new square resolution.
        /// </summary>
        public static float[,] ResampleBilinear(float[,] src, int dstRes)
        {
            int srcRes = src.GetLength(0);
            var dst = new float[dstRes, dstRes];
            float scale = (srcRes - 1f) / Mathf.Max(dstRes - 1f, 1f);

            for (int z = 0; z < dstRes; z++)
            {
                float sz = z * scale;
                int z0 = Mathf.Clamp((int)sz, 0, srcRes - 1);
                int z1 = Mathf.Min(z0 + 1, srcRes - 1);
                float tz = sz - z0;

                for (int x = 0; x < dstRes; x++)
                {
                    float sx = x * scale;
                    int x0 = Mathf.Clamp((int)sx, 0, srcRes - 1);
                    int x1 = Mathf.Min(x0 + 1, srcRes - 1);
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

            return dst;
        }

        /// <summary>
        /// Separable [1, 2, 1] / 4 Gaussian blur pass (fast O(N^2), preserves edges nicely).
        /// </summary>
        public static float[,] GaussianSmoothPass(float[,] src)
        {
            int n = src.GetLength(0);
            var tmp = new float[n, n];
            var dst = new float[n, n];

            // Horizontal pass
            for (int z = 0; z < n; z++)
            {
                tmp[z, 0] = (src[z, 0] * 3f + src[z, 1]) * 0.25f;
                for (int x = 1; x < n - 1; x++)
                {
                    tmp[z, x] = (src[z, x - 1] + src[z, x] * 2f + src[z, x + 1]) * 0.25f;
                }
                tmp[z, n - 1] = (src[z, n - 2] + src[z, n - 1] * 3f) * 0.25f;
            }

            // Vertical pass
            for (int x = 0; x < n; x++)
            {
                dst[0, x] = (tmp[0, x] * 3f + tmp[1, x]) * 0.25f;
                for (int z = 1; z < n - 1; z++)
                {
                    dst[z, x] = (tmp[z - 1, x] + tmp[z, x] * 2f + tmp[z + 1, x]) * 0.25f;
                }
                dst[n - 1, x] = (tmp[n - 2, x] + tmp[n - 1, x] * 3f) * 0.25f;
            }

            return dst;
        }

        /// <summary>
        /// Bake a tangent-space normal map Texture2D from the processed [z, x] heightmap
        /// to add crisp micro-relief shading to URP Terrain Lit layers.
        /// Set <paramref name="packDxt5nm"/> to true when binding directly to Unity runtime URP shaders on desktop/Metal.
        /// </summary>
        public static Texture2D GenerateNormalMap(
            float[,] heights,
            float terrainSize,
            float terrainHeight,
            float strength = 1.5f,
            bool packDxt5nm = false
        )
        {
            int n = heights.GetLength(0);
            var tex = new Texture2D(n, n, TextureFormat.RGBA32, mipChain: true, linear: true)
            {
                name = "DepthTerrain_GeneratedNormalMap",
                wrapMode = TextureWrapMode.Clamp,
                filterMode = FilterMode.Bilinear
            };

            var pixels = new Color32[n * n];
            float cellSize = terrainSize / Mathf.Max(n - 1, 1);
            float scale = (terrainHeight / Mathf.Max(cellSize, 1e-4f)) * Mathf.Max(strength, 0f);

            for (int z = 0; z < n; z++)
            {
                int zm = Mathf.Max(z - 1, 0);
                int zp = Mathf.Min(z + 1, n - 1);
                int rowOff = z * n;

                for (int x = 0; x < n; x++)
                {
                    int xm = Mathf.Max(x - 1, 0);
                    int xp = Mathf.Min(x + 1, n - 1);

                    float dhdx = (heights[z, xp] - heights[z, xm]) * 0.5f * scale;
                    float dhdz = (heights[zp, x] - heights[zm, x]) * 0.5f * scale;

                    var normal = new Vector3(-dhdx, -dhdz, 1f).normalized;
                    byte nx = (byte)Mathf.RoundToInt((normal.x * 0.5f + 0.5f) * 255f);
                    byte ny = (byte)Mathf.RoundToInt((normal.y * 0.5f + 0.5f) * 255f);
                    byte nz = (byte)Mathf.RoundToInt((normal.z * 0.5f + 0.5f) * 255f);
                    pixels[rowOff + x] = packDxt5nm
                        ? new Color32(255, ny, 255, nx)
                        : new Color32(nx, ny, nz, 255);
                }
            }

            tex.SetPixels32(pixels);
            tex.Apply(updateMipmaps: true, makeNoLongerReadable: false);
            return tex;
        }
    }
}
