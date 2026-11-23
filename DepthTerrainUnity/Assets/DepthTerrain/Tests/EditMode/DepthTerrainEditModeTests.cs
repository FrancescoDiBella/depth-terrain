using System.Collections.Generic;
using System.Diagnostics;
using NUnit.Framework;
using UnityEngine;
using Debug = UnityEngine.Debug;

namespace DepthTerrain.Tests
{
    public class DepthTerrainEditModeTests
    {
        [Test]
        public void NearestUnityResolution_SnapsToValidPowerOfTwoPlusOne()
        {
            Assert.AreEqual(33, HeightmapProcessor.NearestUnityResolution(16));
            Assert.AreEqual(129, HeightmapProcessor.NearestUnityResolution(128));
            Assert.AreEqual(257, HeightmapProcessor.NearestUnityResolution(256));
            Assert.AreEqual(513, HeightmapProcessor.NearestUnityResolution(513));
            Assert.AreEqual(513, HeightmapProcessor.NearestUnityResolution(500));
            Assert.AreEqual(1025, HeightmapProcessor.NearestUnityResolution(1024));
        }

        [Test]
        public void EncodeAndDecodeRaw16_PreservesNorthSouthFlipYAlignment()
        {
            const int res = 33;
            var heights = new float[res, res];
            // North-West corner (z = res-1, x = 0) = 1.0, South-East corner (z = 0, x = res-1) = 0.0
            heights[res - 1, 0] = 1.0f;
            heights[0, res - 1] = 0.0f;
            heights[16, 16] = 0.5f;

            byte[] raw = HeightmapProcessor.EncodeRaw16(heights, flipY: true);
            Assert.AreEqual(res * res * 2, raw.Length);

            // First pixel in top-to-bottom RAW file (row=0, col=0) must correspond to North-West (1.0 -> 65535)
            ushort topLeft = (ushort)(raw[0] | (raw[1] << 8));
            Assert.AreEqual(65535, topLeft);

            float[,] decoded = HeightmapProcessor.DecodeRaw16(raw, res, littleEndian: true, flipY: true);
            Assert.AreEqual(1.0f, decoded[res - 1, 0], 1e-4f);
            Assert.AreEqual(0.0f, decoded[0, res - 1], 1e-4f);
            Assert.AreEqual(0.5f, decoded[16, 16], 1e-3f);
        }

        [Test]
        public void ProcessHeights_InvertSmoothAndFalloffBehaveDeterministically()
        {
            const int res = 65;
            var src = new float[res, res];
            for (int z = 0; z < res; z++)
            for (int x = 0; x < res; x++)
                src[z, x] = (z == 32 && x == 32) ? 1.0f : 0.25f;

            var settings = new TerrainBuildSettings
            {
                targetResolution = 65,
                invertHeight = true,
                smoothingPasses = 2,
                elevationGamma = 1.0f,
                edgeFalloff = true,
                edgeFalloffMargin = 0.1f
            };

            float[,] processed = HeightmapProcessor.ProcessHeights(src, settings);
            Assert.AreEqual(65, processed.GetLength(0));
            Assert.AreEqual(65, processed.GetLength(1));
            // Inverted spike at center is lower than surrounding plateau
            Assert.Less(processed[32, 32], processed[20, 20]);
            // Edge falloff zeroes borders
            Assert.AreEqual(0.0f, processed[0, 32], 1e-5f);
            Assert.AreEqual(0.0f, processed[64, 32], 1e-5f);
        }

        [Test]
        public void HeightmapProcessor_ParallelMatchesSequentialEquivalence()
        {
            const int res = 257;
            float[,] src = BuildSyntheticTerrainField(res);
            byte[] raw = HeightmapProcessor.EncodeRaw16(src, flipY: true);

            float[,] decSeq = HeightmapProcessor.DecodeRaw16(raw, res, littleEndian: true, flipY: true, useParallel: false);
            float[,] decPar = HeightmapProcessor.DecodeRaw16(raw, res, littleEndian: true, flipY: true, useParallel: true);
            for (int z = 0; z < res; z++)
            for (int x = 0; x < res; x++)
                Assert.AreEqual(decSeq[z, x], decPar[z, x], 1e-6f);

            var settings = new TerrainBuildSettings
            {
                targetResolution = res,
                invertHeight = true,
                smoothingPasses = 4,
                cliffDetailStrength = 0.5f,
                elevationGamma = 1.25f,
                floorClamp = 0.05f,
                terraceSteps = 8,
                edgeFalloff = true,
                edgeFalloffMargin = 0.12f
            };

            float[,] seq = HeightmapProcessor.ProcessHeights(src, settings, useParallel: false);
            float[,] par = HeightmapProcessor.ProcessHeights(src, settings, useParallel: true);

            for (int z = 0; z < res; z++)
            for (int x = 0; x < res; x++)
                Assert.AreEqual(seq[z, x], par[z, x], 1e-6f, $"Mismatch at [{z},{x}]");

            Texture2D normSeq = HeightmapProcessor.GenerateNormalMap(seq, 500f, 120f, 1.4f, packDxt5nm: true, useParallel: false);
            Texture2D normPar = HeightmapProcessor.GenerateNormalMap(par, 500f, 120f, 1.4f, packDxt5nm: true, useParallel: true);
            try
            {
                Color32[] pxSeq = normSeq.GetPixels32();
                Color32[] pxPar = normPar.GetPixels32();
                Assert.AreEqual(pxSeq.Length, pxPar.Length);
                for (int i = 0; i < pxSeq.Length; i++)
                {
                    Assert.AreEqual(pxSeq[i].r, pxPar[i].r);
                    Assert.AreEqual(pxSeq[i].g, pxPar[i].g);
                    Assert.AreEqual(pxSeq[i].b, pxPar[i].b);
                    Assert.AreEqual(pxSeq[i].a, pxPar[i].a);
                }
            }
            finally
            {
                Object.DestroyImmediate(normSeq);
                Object.DestroyImmediate(normPar);
            }
        }

        [Test]
        public void HeightmapProcessor_PingPongMultiPassSmoothingMatchesReference()
        {
            const int res = 129;
            float[,] src = BuildSyntheticTerrainField(res);

            // Reference: 5 separate GaussianSmoothPass calls
            float[,] expected = (float[,])src.Clone();
            for (int p = 0; p < 5; p++)
                expected = HeightmapProcessor.GaussianSmoothPass(expected, useParallel: false);

            // Ping-pong in-place with a single scratch buffer
            float[,] actual = (float[,])src.Clone();
            var scratch = new float[res, res];
            HeightmapProcessor.GaussianSmoothInPlace(actual, passes: 5, scratch: scratch, useParallel: true);

            for (int z = 0; z < res; z++)
            for (int x = 0; x < res; x++)
                Assert.AreEqual(expected[z, x], actual[z, x], 1e-6f);
        }

        [Test]
        public void GenerateNormalMap_ProducesValidTangentSpaceNormals()
        {
            const int res = 33;
            var flat = new float[res, res];
            for (int z = 0; z < res; z++)
            for (int x = 0; x < res; x++)
                flat[z, x] = 0.5f;

            Texture2D normalMap = HeightmapProcessor.GenerateNormalMap(flat, 500f, 100f, 1.5f);
            try
            {
                Assert.AreEqual(res, normalMap.width);
                Assert.AreEqual(res, normalMap.height);
                // Flat terrain normal is (0, 0, 1) -> RGB ~(128, 128, 255)
                Color center = normalMap.GetPixel(16, 16);
                Assert.AreEqual(0.5f, center.r, 0.02f);
                Assert.AreEqual(0.5f, center.g, 0.02f);
                Assert.Greater(center.b, 0.98f);
            }
            finally
            {
                Object.DestroyImmediate(normalMap);
            }
        }

        [Test]
        public void BuildOrUpdateTerrain_CreatesValidTerrainColliderLayerAndWater()
        {
            const int res = 65;
            var heights = new float[res, res];
            for (int z = 0; z < res; z++)
            for (int x = 0; x < res; x++)
                heights[z, x] = (float)z / (res - 1);

            var albedo = new Texture2D(64, 64, TextureFormat.RGBA32, false);
            var root = new GameObject("TestDepthTerrainRoot");
            var settings = new TerrainBuildSettings
            {
                terrainSize = 400f,
                terrainHeight = 100f,
                targetResolution = res,
                showWaterPlane = true,
                waterLevel = 0.2f,
                normalMapStrength = 1.0f
            };

            try
            {
                var output = DepthTerrainBuilder.BuildOrUpdateTerrain(null, root.transform, heights, albedo, settings);
                Assert.IsNotNull(output.Terrain);
                Assert.IsNotNull(output.TerrainData);
                Assert.IsNotNull(output.WaterPlane);
                Assert.AreEqual(res, output.TerrainData.heightmapResolution);
                Assert.AreEqual(new Vector3(400f, 100f, 400f), output.TerrainData.size);
                Assert.AreEqual(1, output.TerrainData.terrainLayers.Length);
                Assert.AreSame(albedo, output.TerrainData.terrainLayers[0].diffuseTexture);
                Assert.IsTrue(output.WaterPlane.activeSelf);
                Assert.AreEqual(20f, output.WaterPlane.transform.localPosition.y, 0.01f);
                Assert.Greater(output.Timings.TotalBuildMs, 0.0);
                Assert.AreEqual(res, output.Timings.Resolution);

                if (output.NormalMap != null) Object.DestroyImmediate(output.NormalMap);
                if (output.TerrainLayer != null) Object.DestroyImmediate(output.TerrainLayer);
                if (output.TerrainData != null) Object.DestroyImmediate(output.TerrainData);
            }
            finally
            {
                Object.DestroyImmediate(root);
                Object.DestroyImmediate(albedo);
            }
        }

        [Test]
        public void BuildOrUpdateTerrain_ReusesAndDisposesNormalMapsWithoutLeaks()
        {
            int baselineNormalCount = CountNamedTextures(DepthTerrainBuilder.GeneratedNormalMapName);

            var albedo = new Texture2D(64, 64, TextureFormat.RGBA32, false);
            var root = new GameObject("TestLeakFreeTerrainRoot");
            var heights65 = BuildSyntheticTerrainField(65);
            var heights129 = BuildSyntheticTerrainField(129);
            var settings = new TerrainBuildSettings
            {
                terrainSize = 400f,
                terrainHeight = 100f,
                targetResolution = 65,
                normalMapStrength = 1.0f
            };

            DepthTerrainBuilder.BuildOutput output = default;
            try
            {
                // 1) Initial build + 11 repeated slider-like rebuilds at 65x65 must reuse the exact same NormalMap & TerrainLayer
                output = DepthTerrainBuilder.BuildOrUpdateTerrain(null, root.transform, heights65, albedo, settings);
                Texture2D firstNormal = output.NormalMap;
                TerrainLayer firstLayer = output.TerrainLayer;
                Assert.IsNotNull(firstNormal);
                Assert.AreEqual(baselineNormalCount + 1, CountNamedTextures(DepthTerrainBuilder.GeneratedNormalMapName));

                for (int i = 0; i < 11; i++)
                {
                    settings.terrainHeight = 80f + i * 5f;
                    output = DepthTerrainBuilder.BuildOrUpdateTerrain(
                        output.Terrain,
                        root.transform,
                        heights65,
                        albedo,
                        settings,
                        output.WaterPlane
                    );
                    Assert.AreSame(firstNormal, output.NormalMap, "NormalMap should be reused in-place at matching resolution.");
                    Assert.AreSame(firstLayer, output.TerrainLayer, "TerrainLayer should be reused in-place.");
                }
                Assert.AreEqual(baselineNormalCount + 1, CountNamedTextures(DepthTerrainBuilder.GeneratedNormalMapName));

                // 2) Resolution change to 129x129 must dispose old 65x65 normal map and allocate exactly 1 new one
                settings.targetResolution = 129;
                output = DepthTerrainBuilder.BuildOrUpdateTerrain(
                    output.Terrain,
                    root.transform,
                    heights129,
                    albedo,
                    settings,
                    output.WaterPlane
                );
                Assert.AreEqual(129, output.NormalMap.width);
                Assert.AreEqual(baselineNormalCount + 1, CountNamedTextures(DepthTerrainBuilder.GeneratedNormalMapName));

                // 3) Disabling normal map must dispose it cleanly
                settings.normalMapStrength = 0f;
                output = DepthTerrainBuilder.BuildOrUpdateTerrain(
                    output.Terrain,
                    root.transform,
                    heights129,
                    albedo,
                    settings,
                    output.WaterPlane
                );
                Assert.IsNull(output.NormalMap);
                Assert.AreEqual(baselineNormalCount, CountNamedTextures(DepthTerrainBuilder.GeneratedNormalMapName));

                if (output.TerrainLayer != null) Object.DestroyImmediate(output.TerrainLayer);
                if (output.TerrainData != null) Object.DestroyImmediate(output.TerrainData);
            }
            finally
            {
                Object.DestroyImmediate(root);
                Object.DestroyImmediate(albedo);
            }
        }

        [Test]
        public void BuildOrUpdateTerrain_TopographicMasksAndCliffDetailWorkCleanly()
        {
            int baselineMaskCount = CountNamedTextures(DepthTerrainBuilder.DerivedTopographicMaskName);
            const int res = 65;
            float[,] raw = BuildSyntheticTerrainField(res);

            var settingsNoCliff = new TerrainBuildSettings
            {
                targetResolution = res,
                smoothingPasses = 1,
                cliffDetailStrength = 0f,
                useTerrainMasks = true,
                normalMapStrength = 0.5f
            };
            var settingsCliff = settingsNoCliff.Clone();
            settingsCliff.cliffDetailStrength = 1.0f;

            float[,] baseH = HeightmapProcessor.ProcessHeights(raw, settingsNoCliff);
            float[,] cliffH = HeightmapProcessor.ProcessHeights(raw, settingsCliff);

            bool anyDiff = false;
            for (int z = 0; z < res; z++)
            for (int x = 0; x < res; x++)
            {
                Assert.GreaterOrEqual(cliffH[z, x], 0f);
                Assert.LessOrEqual(cliffH[z, x], 1f);
                if (Mathf.Abs(cliffH[z, x] - baseH[z, x]) > 1e-4f)
                    anyDiff = true;
            }
            Assert.IsTrue(anyDiff, "cliffDetailStrength > 0 should enhance cliff/ridge contrast.");

            // Synthetic terrain_masks.png: R=slope(200), G=AO(120), B=curvature(180), A=255
            var sourceMasks = new Texture2D(16, 16, TextureFormat.RGBA32, false, linear: true);
            var maskPixels = new Color32[16 * 16];
            for (int i = 0; i < maskPixels.Length; i++)
                maskPixels[i] = new Color32(200, 120, 180, 255);
            sourceMasks.SetPixels32(maskPixels);
            sourceMasks.Apply();

            var albedo = new Texture2D(16, 16, TextureFormat.RGBA32, false);
            var root = new GameObject("TestTopoMasksRoot");
            try
            {
                var output = DepthTerrainBuilder.BuildOrUpdateTerrain(
                    null,
                    root.transform,
                    cliffH,
                    albedo,
                    settingsCliff,
                    null,
                    sourceMasks
                );
                Assert.IsNotNull(output.MaskMap);
                Assert.AreEqual(DepthTerrainBuilder.DerivedTopographicMaskName, output.MaskMap.name);
                Assert.AreEqual(baselineMaskCount + 1, CountNamedTextures(DepthTerrainBuilder.DerivedTopographicMaskName));

                Color32 packed = output.MaskMap.GetPixels32()[0];
                Assert.AreEqual(0, packed.r, "URP Terrain Metallic channel (R) must remain matte dielectric 0.");
                Assert.Less(packed.g, 240, "Topographic AO (G) should attenuate crevices.");
                Assert.LessOrEqual(packed.a, 25, "Smoothness (A) must stay matte (<= ~0.1).");

                // Toggling useTerrainMasks off must dispose the derived mask texture cleanly
                settingsCliff.useTerrainMasks = false;
                output = DepthTerrainBuilder.BuildOrUpdateTerrain(
                    output.Terrain,
                    root.transform,
                    cliffH,
                    albedo,
                    settingsCliff,
                    output.WaterPlane,
                    sourceMasks
                );
                Assert.AreEqual(baselineMaskCount, CountNamedTextures(DepthTerrainBuilder.DerivedTopographicMaskName));

                if (output.NormalMap != null) Object.DestroyImmediate(output.NormalMap);
                if (output.TerrainLayer != null) Object.DestroyImmediate(output.TerrainLayer);
                if (output.TerrainData != null) Object.DestroyImmediate(output.TerrainData);
            }
            finally
            {
                Object.DestroyImmediate(root);
                Object.DestroyImmediate(albedo);
                Object.DestroyImmediate(sourceMasks);
            }
        }

        [Test]
        public void DepthPipelineBridge_BuildsSafeCliArgumentsForAllModes()
        {
            var mapReq = new PipelineRunRequest
            {
                sourceMode = PipelineSourceMode.RealMap,
                placeQuery = "Mount Vesuvius, Italy",
                useCoordinates = true,
                latitude = 40.8218f,
                longitude = 14.4260f,
                zoom = 14,
                tileGrid = 3,
                model = DepthPro2ModelVariant.Base_B,
                topDownMode = TopDownAdaptationMode.MultiscaleTTA
            };

            List<string> mapArgs = DepthPipelineBridge.BuildArguments(mapReq, "/tmp/out_map");
            CollectionAssert.Contains(mapArgs, "from-map");
            CollectionAssert.Contains(mapArgs, "depthpro2-b");
            CollectionAssert.Contains(mapArgs, "multiscale_tta");
            CollectionAssert.Contains(mapArgs, "--no-guided-filter");
            CollectionAssert.DoesNotContain(mapArgs, "--export-ply");
            CollectionAssert.Contains(mapArgs, "/tmp/out_map");

            var fictionalReq = new PipelineRunRequest
            {
                sourceMode = PipelineSourceMode.FictionalWorld,
                fictionalBiome = "volcanic_island",
                fictionalSeed = 42,
                model = DepthPro2ModelVariant.Small_S,
                topDownMode = TopDownAdaptationMode.FastMultiscale,
                useGuidedFilter = true,
                exportPointCloudPly = true
            };

            List<string> ficArgs = DepthPipelineBridge.BuildArguments(fictionalReq, "/tmp/out_fic");
            CollectionAssert.Contains(ficArgs, "from-fictional");
            CollectionAssert.Contains(ficArgs, "volcanic_island");
            CollectionAssert.Contains(ficArgs, "depthpro2-s");
            CollectionAssert.Contains(ficArgs, "fast_multiscale");
            CollectionAssert.Contains(ficArgs, "--guided-filter");
            CollectionAssert.Contains(ficArgs, "--export-ply");
        }

        [Test]
        public void DepthPipelineBridge_BuildsWarmServerEndpointsAndJsonPayloads()
        {
            Assert.AreEqual(
                "http://127.0.0.1:8765/infer/map",
                DepthPipelineBridge.BuildWarmServerEndpoint(PipelineSourceMode.RealMap)
            );
            Assert.AreEqual(
                "http://127.0.0.1:8765/infer/image",
                DepthPipelineBridge.BuildWarmServerEndpoint(PipelineSourceMode.LocalImage)
            );
            Assert.AreEqual(
                "http://127.0.0.1:8765/infer/fictional",
                DepthPipelineBridge.BuildWarmServerEndpoint(PipelineSourceMode.FictionalWorld)
            );
            Assert.AreEqual(
                "http://127.0.0.1:8765/infer/video",
                DepthPipelineBridge.BuildWarmServerEndpoint(PipelineSourceMode.AerialVideo)
            );

            var req = new PipelineRunRequest
            {
                sourceMode = PipelineSourceMode.FictionalWorld,
                fictionalBiome = "alpine_peaks",
                fictionalSeed = 77,
                model = DepthPro2ModelVariant.Base_B,
                topDownMode = TopDownAdaptationMode.FastMultiscale,
                useGuidedFilter = true,
                exportPointCloudPly = true,
                heightmapResolution = 513
            };

            string json = DepthPipelineBridge.BuildWarmServerRequestJson(req, "/tmp/warm_out");
            StringAssert.Contains("\"outputDir\":\"/tmp/warm_out\"", json);
            StringAssert.Contains("\"biome\":\"alpine_peaks\"", json);
            StringAssert.Contains("\"topDownMode\":\"fast_multiscale\"", json);
            StringAssert.Contains("\"guidedFilter\":true", json);
            StringAssert.Contains("\"exportPly\":true", json);
            StringAssert.Contains("\"model\":\"depthpro2-b\"", json);
        }

        [Test]
        public void Benchmark_HeightmapProcessingAndNormalMap_513And1025()
        {
            RunResolutionBenchmark(513);
            RunResolutionBenchmark(1025);
        }

        private static void RunResolutionBenchmark(int res)
        {
            float[,] src = BuildSyntheticTerrainField(res);
            byte[] raw = HeightmapProcessor.EncodeRaw16(src, flipY: true);
            var settings = new TerrainBuildSettings
            {
                terrainSize = 500f,
                terrainHeight = 125f,
                targetResolution = res,
                smoothingPasses = 3,
                elevationGamma = 1.15f,
                edgeFalloff = true,
                normalMapStrength = 1.0f
            };

            // Warm up JIT & thread pool
            HeightmapProcessor.DecodeRaw16(raw, res, useParallel: true);
            float[,] warmH = HeightmapProcessor.ProcessHeights(src, settings, useParallel: true);
            Texture2D warmNorm = HeightmapProcessor.GenerateNormalMap(warmH, 500f, 125f, 1.0f, packDxt5nm: true, useParallel: true);

            const int iters = 3;

            var sw = Stopwatch.StartNew();
            for (int i = 0; i < iters; i++)
                HeightmapProcessor.DecodeRaw16(raw, res, useParallel: false);
            double decodeSeqMs = sw.Elapsed.TotalMilliseconds / iters;

            sw.Restart();
            for (int i = 0; i < iters; i++)
                HeightmapProcessor.DecodeRaw16(raw, res, useParallel: true);
            double decodeParMs = sw.Elapsed.TotalMilliseconds / iters;

            sw.Restart();
            for (int i = 0; i < iters; i++)
                HeightmapProcessor.ProcessHeights(src, settings, useParallel: false);
            double procSeqMs = sw.Elapsed.TotalMilliseconds / iters;

            float[,] processed = null;
            sw.Restart();
            for (int i = 0; i < iters; i++)
                processed = HeightmapProcessor.ProcessHeights(src, settings, useParallel: true);
            double procParMs = sw.Elapsed.TotalMilliseconds / iters;

            sw.Restart();
            for (int i = 0; i < iters; i++)
                HeightmapProcessor.GenerateNormalMap(processed, 500f, 125f, 1.0f, packDxt5nm: true, reuseTexture: warmNorm, useParallel: false);
            double normSeqMs = sw.Elapsed.TotalMilliseconds / iters;

            sw.Restart();
            for (int i = 0; i < iters; i++)
                HeightmapProcessor.GenerateNormalMap(processed, 500f, 125f, 1.0f, packDxt5nm: true, reuseTexture: warmNorm, useParallel: true);
            double normParMs = sw.Elapsed.TotalMilliseconds / iters;

            Object.DestroyImmediate(warmNorm);

            var albedo = new Texture2D(64, 64, TextureFormat.RGBA32, false);
            var root = new GameObject($"BenchmarkTerrain_{res}");
            try
            {
                // Initial build + warm rebuild (measuring live slider-drag rebuild cost)
                var first = DepthTerrainBuilder.BuildOrUpdateTerrain(null, root.transform, processed, albedo, settings);
                var rebuild = DepthTerrainBuilder.BuildOrUpdateTerrain(
                    first.Terrain,
                    root.transform,
                    processed,
                    albedo,
                    settings,
                    first.WaterPlane
                );

                Debug.Log(
                    $"[DepthTerrain Benchmark {res}x{res}] " +
                    $"Decode: {decodeSeqMs:F2}ms -> {decodeParMs:F2}ms ({decodeSeqMs / System.Math.Max(decodeParMs, 0.01):F2}x) | " +
                    $"Process(3-pass): {procSeqMs:F2}ms -> {procParMs:F2}ms ({procSeqMs / System.Math.Max(procParMs, 0.01):F2}x) | " +
                    $"NormalMap(DXT5nm): {normSeqMs:F2}ms -> {normParMs:F2}ms ({normSeqMs / System.Math.Max(normParMs, 0.01):F2}x) | " +
                    $"FullRebuild: {rebuild.Timings.TotalBuildMs:F2}ms (Norm {rebuild.Timings.NormalMapMs:F2}ms, Upload {rebuild.Timings.TerrainUploadMs:F2}ms)"
                );

                Assert.Greater(procParMs, 0.0);
                Assert.Greater(normParMs, 0.0);
                Assert.Greater(rebuild.Timings.TotalBuildMs, 0.0);

                if (rebuild.NormalMap != null) Object.DestroyImmediate(rebuild.NormalMap);
                if (rebuild.TerrainLayer != null) Object.DestroyImmediate(rebuild.TerrainLayer);
                if (rebuild.TerrainData != null) Object.DestroyImmediate(rebuild.TerrainData);
            }
            finally
            {
                Object.DestroyImmediate(root);
                Object.DestroyImmediate(albedo);
            }
        }

        private static float[,] BuildSyntheticTerrainField(int res)
        {
            var heights = new float[res, res];
            float inv = 1f / (res - 1);
            for (int z = 0; z < res; z++)
            {
                float vz = z * inv;
                for (int x = 0; x < res; x++)
                {
                    float vx = x * inv;
                    float dx = vx - 0.5f;
                    float dz = vz - 0.5f;
                    float r = Mathf.Sqrt(dx * dx + dz * dz);
                    float cone = Mathf.Clamp01(1f - r * 2.1f);
                    float ridges = 0.15f * Mathf.Sin(vx * 22f) * Mathf.Cos(vz * 19f) + 0.08f * Mathf.Sin((vx + vz) * 45f);
                    heights[z, x] = Mathf.Clamp01(cone * 0.8f + 0.15f + ridges);
                }
            }
            return heights;
        }

        private static int CountNamedTextures(string texName)
        {
            int count = 0;
            Texture2D[] all = Resources.FindObjectsOfTypeAll<Texture2D>();
            for (int i = 0; i < all.Length; i++)
            {
                if (all[i] != null && all[i].name == texName)
                    count++;
            }
            return count;
        }
    }
}
