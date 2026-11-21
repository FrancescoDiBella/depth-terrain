using System;
using UnityEngine;
using UnityEngine.Rendering;

namespace DepthTerrain
{
    /// <summary>
    /// Constructs and updates a Unity Terrain, TerrainData, URP TerrainLayer (satellite/fictional albedo +
    /// height-derived normal map), TerrainCollider, and dynamic translucent URP Water Plane.
    /// </summary>
    public static class DepthTerrainBuilder
    {
        public const string DefaultTerrainObjectName = "DepthPro2_Terrain";
        public const string DefaultWaterObjectName = "DepthPro2_WaterPlane";

        public struct BuildOutput
        {
            public Terrain Terrain;
            public TerrainData TerrainData;
            public TerrainLayer TerrainLayer;
            public Texture2D NormalMap;
            public GameObject WaterPlane;
            public Bounds WorldBounds;
        }

        /// <summary>
        /// Build or update a Unity Terrain in the scene from a processed [z, x] heightmap and top-down albedo texture.
        /// </summary>
        public static BuildOutput BuildOrUpdateTerrain(
            Terrain existingTerrain,
            Transform parent,
            float[,] processedHeights,
            Texture2D albedoTexture,
            TerrainBuildSettings settings,
            GameObject existingWaterPlane = null
        )
        {
            if (processedHeights == null)
                throw new ArgumentNullException(nameof(processedHeights));
            if (settings == null)
                throw new ArgumentNullException(nameof(settings));

            int res = processedHeights.GetLength(0);
            int validRes = HeightmapProcessor.NearestUnityResolution(res);
            if (validRes != res)
            {
                processedHeights = HeightmapProcessor.ResampleBilinear(processedHeights, validRes);
                res = validRes;
            }

            Terrain terrain = existingTerrain;
            TerrainData terrainData;
            if (terrain == null)
            {
                var go = new GameObject(DefaultTerrainObjectName);
                if (parent != null) go.transform.SetParent(parent, false);
                terrain = go.AddComponent<Terrain>();
                var col = go.AddComponent<TerrainCollider>();
                terrainData = new TerrainData { name = "DepthPro2_TerrainData" };
                terrain.terrainData = terrainData;
                col.terrainData = terrainData;
            }
            else
            {
                terrainData = terrain.terrainData;
                if (terrainData == null)
                {
                    terrainData = new TerrainData { name = "DepthPro2_TerrainData" };
                    terrain.terrainData = terrainData;
                }
                var col = terrain.GetComponent<TerrainCollider>();
                if (col == null) col = terrain.gameObject.AddComponent<TerrainCollider>();
                col.terrainData = terrainData;
            }

            // Ensure URP Terrain Lit material is assigned when running under URP
            if (terrain.materialTemplate == null)
            {
                var urpTerrainShader = Shader.Find("Universal Render Pipeline/Terrain/Lit");
                if (urpTerrainShader != null)
                {
                    var mat = new Material(urpTerrainShader) { name = "DepthPro2_TerrainMaterial" };
                    terrain.materialTemplate = mat;
                }
            }

            // Order matters in Unity: set heightmapResolution before size and heights
            terrainData.heightmapResolution = res;
            terrainData.size = new Vector3(settings.terrainSize, settings.terrainHeight, settings.terrainSize);
            terrainData.SetHeights(0, 0, processedHeights);

            // Center terrain horizontally around world origin (0, 0, 0) for intuitive orbit camera framing
            float half = settings.terrainSize * 0.5f;
            terrain.transform.localPosition = new Vector3(-half, 0f, -half);
            terrain.heightmapPixelError = 2f;
            terrain.basemapDistance = Mathf.Max(2000f, settings.terrainSize * 3f);
            terrain.drawInstanced = true;

            // Generate procedural normal map from heightfield for crisp relief micro-shading
            Texture2D normalMap = null;
            if (settings.normalMapStrength > 0.001f)
            {
                normalMap = HeightmapProcessor.GenerateNormalMap(
                    processedHeights,
                    settings.terrainSize,
                    settings.terrainHeight,
                    settings.normalMapStrength
                );
            }

            // Configure single full-footprint TerrainLayer mapping top-down albedo 1:1 to terrain bounds
            var layer = new TerrainLayer
            {
                name = "DepthPro2_AlbedoLayer",
                diffuseTexture = albedoTexture,
                normalMapTexture = normalMap,
                normalScale = settings.normalMapStrength,
                tileSize = new Vector2(settings.terrainSize, settings.terrainSize),
                tileOffset = Vector2.zero,
                smoothness = 0.14f,
                metallic = 0.0f
            };

            terrainData.alphamapResolution = 64;
            terrainData.terrainLayers = new[] { layer };
            terrain.Flush();

            // Create or update translucent URP water plane
            GameObject waterPlane = UpdateWaterPlane(existingWaterPlane, parent, settings);

            var bounds = new Bounds(
                new Vector3(0f, settings.terrainHeight * 0.45f, 0f),
                new Vector3(settings.terrainSize, settings.terrainHeight, settings.terrainSize)
            );

            return new BuildOutput
            {
                Terrain = terrain,
                TerrainData = terrainData,
                TerrainLayer = layer,
                NormalMap = normalMap,
                WaterPlane = waterPlane,
                WorldBounds = bounds
            };
        }

        public static GameObject UpdateWaterPlane(GameObject existingWater, Transform parent, TerrainBuildSettings settings)
        {
            GameObject water = existingWater;
            if (water == null)
            {
                water = GameObject.CreatePrimitive(PrimitiveType.Plane);
                water.name = DefaultWaterObjectName;
                if (parent != null) water.transform.SetParent(parent, false);

                var col = water.GetComponent<Collider>();
                if (col != null)
                {
                    if (Application.isPlaying) UnityEngine.Object.Destroy(col);
                    else UnityEngine.Object.DestroyImmediate(col);
                }

                var renderer = water.GetComponent<MeshRenderer>();
                if (renderer != null)
                {
                    renderer.sharedMaterial = CreateUrpWaterMaterial();
                    renderer.shadowCastingMode = ShadowCastingMode.Off;
                    renderer.receiveShadows = true;
                }
            }

            water.SetActive(settings.showWaterPlane && settings.waterLevel > 0.001f);
            // Built-in Unity Plane is 10x10 units
            float scaleXZ = (settings.terrainSize / 10f) * 1.005f;
            float waterY = settings.waterLevel * settings.terrainHeight;
            water.transform.localPosition = new Vector3(0f, waterY, 0f);
            water.transform.localScale = new Vector3(scaleXZ, 1f, scaleXZ);
            return water;
        }

        public static Material CreateUrpWaterMaterial()
        {
            Shader shader = Shader.Find("Universal Render Pipeline/Lit") ?? Shader.Find("Standard");
            var mat = new Material(shader) { name = "DepthPro2_WaterMaterial" };

            if (shader != null && shader.name.Contains("Universal Render Pipeline"))
            {
                mat.SetFloat("_Surface", 1f); // Transparent
                mat.SetFloat("_Blend", 0f);   // Alpha
                mat.SetInt("_SrcBlend", (int)BlendMode.SrcAlpha);
                mat.SetInt("_DstBlend", (int)BlendMode.OneMinusSrcAlpha);
                mat.SetInt("_ZWrite", 0);
                mat.SetFloat("_Smoothness", 0.92f);
                mat.SetFloat("_Metallic", 0.12f);
                mat.SetColor("_BaseColor", new Color(0.05f, 0.28f, 0.44f, 0.76f));
                mat.EnableKeyword("_SURFACE_TYPE_TRANSPARENT");
                mat.SetOverrideTag("RenderType", "Transparent");
                mat.renderQueue = (int)RenderQueue.Transparent;
            }
            else
            {
                mat.color = new Color(0.05f, 0.28f, 0.44f, 0.76f);
            }

            return mat;
        }
    }
}
