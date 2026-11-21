using System.Collections.Generic;
using NUnit.Framework;
using UnityEngine;

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
            CollectionAssert.Contains(mapArgs, "/tmp/out_map");

            var fictionalReq = new PipelineRunRequest
            {
                sourceMode = PipelineSourceMode.FictionalWorld,
                fictionalBiome = "volcanic_island",
                fictionalSeed = 42,
                model = DepthPro2ModelVariant.Small_S
            };

            List<string> ficArgs = DepthPipelineBridge.BuildArguments(fictionalReq, "/tmp/out_fic");
            CollectionAssert.Contains(ficArgs, "from-fictional");
            CollectionAssert.Contains(ficArgs, "volcanic_island");
            CollectionAssert.Contains(ficArgs, "depthpro2-s");
        }
    }
}
