using System;
using System.Diagnostics;
using System.Runtime.CompilerServices;
using UnityEngine;
using UnityEngine.Rendering;

namespace DepthTerrain
{
    /// <summary>
    /// Constructs and updates a Unity Terrain, TerrainData, URP TerrainLayer (satellite/fictional albedo +
    /// height-derived normal map + optional topographic mask map), TerrainCollider, and dynamic translucent URP Water Plane.
    /// Reuses and properly disposes temporary runtime textures and TerrainLayers to prevent GPU memory leaks.
    /// </summary>
    public static class DepthTerrainBuilder
    {
        public const string DefaultTerrainObjectName = "DepthPro2_Terrain";
        public const string DefaultWaterObjectName = "DepthPro2_WaterPlane";
        public const string RuntimeLayerName = "DepthPro2_AlbedoLayer";
        public const string GeneratedNormalMapName = "DepthTerrain_GeneratedNormalMap";
        public const string DerivedTopographicMaskName = "DepthPro2_DerivedTopographicMaskMap";

        private sealed class DerivedMaskStamp
        {
            public Texture2D SourceMask;
            public float CliffBoost;
        }

        private static Texture2D _sharedMatteMask;
        private static readonly ConditionalWeakTable<Texture2D, DerivedMaskStamp> DerivedMaskStampCache =
            new ConditionalWeakTable<Texture2D, DerivedMaskStamp>();

        public struct BuildOutput
        {
            public Terrain Terrain;
            public TerrainData TerrainData;
            public TerrainLayer TerrainLayer;
            public Texture2D NormalMap;
            public Texture2D MaskMap;
            public GameObject WaterPlane;
            public Bounds WorldBounds;
            public TerrainBuildTimings Timings;
        }

        /// <summary>
        /// Safely destroy a runtime-created UnityEngine.Object in both EditMode and PlayMode.
        /// </summary>
        public static void DestroyObjectSafe(UnityEngine.Object obj)
        {
            if (obj == null) return;
            if (Application.isPlaying) UnityEngine.Object.Destroy(obj);
            else UnityEngine.Object.DestroyImmediate(obj);
        }

        /// <summary>
        /// Build or update a Unity Terrain in the scene from a processed [z, x] heightmap, top-down albedo texture,
        /// and optional Depth Pro 2 topographic masks texture (R=slope, G=AO, B=ridge_curvature).
        /// </summary>
        public static BuildOutput BuildOrUpdateTerrain(
            Terrain existingTerrain,
            Transform parent,
            float[,] processedHeights,
            Texture2D albedoTexture,
            TerrainBuildSettings settings,
            GameObject existingWaterPlane = null,
            Texture2D terrainMasksTexture = null
        )
        {
            if (processedHeights == null)
                throw new ArgumentNullException(nameof(processedHeights));
            if (settings == null)
                throw new ArgumentNullException(nameof(settings));

            var totalSw = Stopwatch.StartNew();

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

            // Inspect existing runtime layer and textures so we reuse in-place or dispose cleanly (zero GPU leaks)
            TerrainLayer reusableLayer = null;
            Texture2D existingNormalMap = null;
            Texture2D existingDerivedMask = null;

            TerrainLayer[] existingLayers = terrainData.terrainLayers;
            if (existingLayers != null && existingLayers.Length > 0)
            {
                for (int i = 0; i < existingLayers.Length; i++)
                {
                    TerrainLayer candidate = existingLayers[i];
                    if (candidate == null) continue;

                    if (reusableLayer == null && candidate.name == RuntimeLayerName)
                    {
                        reusableLayer = candidate;
                        if (candidate.normalMapTexture != null && candidate.normalMapTexture.name == GeneratedNormalMapName)
                            existingNormalMap = candidate.normalMapTexture;
                        if (candidate.maskMapTexture != null &&
                            candidate.maskMapTexture.name == DerivedTopographicMaskName &&
                            candidate.maskMapTexture != _sharedMatteMask)
                            existingDerivedMask = candidate.maskMapTexture;
                    }
                    else if (candidate.name == RuntimeLayerName)
                    {
                        if (candidate.normalMapTexture != null && candidate.normalMapTexture.name == GeneratedNormalMapName)
                            DestroyObjectSafe(candidate.normalMapTexture);
                        if (candidate.maskMapTexture != null &&
                            candidate.maskMapTexture.name == DerivedTopographicMaskName &&
                            candidate.maskMapTexture != _sharedMatteMask)
                        {
                            DerivedMaskStampCache.Remove(candidate.maskMapTexture);
                            DestroyObjectSafe(candidate.maskMapTexture);
                        }
                        DestroyObjectSafe(candidate);
                    }
                }
            }

            // Order matters in Unity: set heightmapResolution before size and heights
            var uploadSw = Stopwatch.StartNew();
            if (terrainData.heightmapResolution != res)
                terrainData.heightmapResolution = res;
            terrainData.size = new Vector3(settings.terrainSize, settings.terrainHeight, settings.terrainSize);
            terrainData.SetHeights(0, 0, processedHeights);
            uploadSw.Stop();

            // Center terrain horizontally around world origin (0, 0, 0) for intuitive orbit camera framing
            float half = settings.terrainSize * 0.5f;
            terrain.transform.localPosition = new Vector3(-half, 0f, -half);
            terrain.heightmapPixelError = 2f;
            terrain.basemapDistance = Mathf.Max(2000f, settings.terrainSize * 3f);
            terrain.drawInstanced = true;

            // Generate or reuse procedural normal map from heightfield (DXT5nm packed for URP runtime Terrain Lit)
            var normalSw = Stopwatch.StartNew();
            Texture2D normalMap = null;
            if (settings.normalMapStrength > 0.001f)
            {
                bool canReuseNormal = existingNormalMap != null &&
                                      existingNormalMap.width == res &&
                                      existingNormalMap.height == res &&
                                      existingNormalMap.format == TextureFormat.RGBA32;

                if (!canReuseNormal && existingNormalMap != null)
                {
                    DestroyObjectSafe(existingNormalMap);
                    existingNormalMap = null;
                }

                normalMap = HeightmapProcessor.GenerateNormalMap(
                    processedHeights,
                    settings.terrainSize,
                    settings.terrainHeight,
                    settings.normalMapStrength,
                    packDxt5nm: true,
                    reuseTexture: existingNormalMap
                );
            }
            else if (existingNormalMap != null)
            {
                DestroyObjectSafe(existingNormalMap);
            }
            normalSw.Stop();

            // Build or reuse URP Terrain MaskMap (keeps matte dielectric shading while optionally applying Depth Pro 2 AO/slope)
            Texture2D maskMap;
            if (terrainMasksTexture != null && settings.useTerrainMasks && terrainMasksTexture.isReadable)
            {
                maskMap = BuildOrUpdateTopographicMaskMap(
                    terrainMasksTexture,
                    existingDerivedMask,
                    settings.cliffDetailStrength
                );
            }
            else
            {
                if (existingDerivedMask != null)
                {
                    DerivedMaskStampCache.Remove(existingDerivedMask);
                    DestroyObjectSafe(existingDerivedMask);
                }
                maskMap = CreateMatteTerrainMaskTexture(smoothness: 0.08f);
            }

            // Configure single full-footprint TerrainLayer mapping top-down albedo 1:1 to terrain bounds
            TerrainLayer layer = reusableLayer != null
                ? reusableLayer
                : new TerrainLayer { name = RuntimeLayerName };

            layer.diffuseTexture = albedoTexture;
            layer.normalMapTexture = normalMap;
            layer.maskMapTexture = maskMap;
            layer.normalScale = settings.normalMapStrength;
            layer.tileSize = new Vector2(settings.terrainSize, settings.terrainSize);
            layer.tileOffset = Vector2.zero;
            layer.smoothness = 0.08f;
            layer.metallic = 0.0f;
            layer.specular = new Color(0.04f, 0.04f, 0.04f, 1f);

            uploadSw.Start();
            if (terrainData.alphamapResolution != 64)
                terrainData.alphamapResolution = 64;
            terrainData.terrainLayers = new[] { layer };
            terrain.Flush();
            uploadSw.Stop();

            // Create or update translucent URP water plane
            GameObject waterPlane = UpdateWaterPlane(existingWaterPlane, parent, settings);

            var bounds = new Bounds(
                new Vector3(0f, settings.terrainHeight * 0.45f, 0f),
                new Vector3(settings.terrainSize, settings.terrainHeight, settings.terrainSize)
            );

            totalSw.Stop();

            var timings = new TerrainBuildTimings
            {
                DecodeMs = 0.0,
                ProcessMs = 0.0,
                NormalMapMs = normalSw.Elapsed.TotalMilliseconds,
                TerrainUploadMs = uploadSw.Elapsed.TotalMilliseconds,
                TotalBuildMs = totalSw.Elapsed.TotalMilliseconds,
                Resolution = res
            };

            return new BuildOutput
            {
                Terrain = terrain,
                TerrainData = terrainData,
                TerrainLayer = layer,
                NormalMap = normalMap,
                MaskMap = maskMap,
                WaterPlane = waterPlane,
                WorldBounds = bounds,
                Timings = timings
            };
        }

        /// <summary>
        /// Convert Depth Pro 2's terrain_masks.png (R=slope, G=AO, B=ridge_curvature, A=255) into a
        /// matte URP Terrain/Lit MaskMap (R=Metallic 0, G=subtle topographic AO, B=curvature, A=matte roughness).
        /// Reuses <paramref name="existingMask"/> in-place and skips redundant uploads when unchanged.
        /// </summary>
        private static Texture2D BuildOrUpdateTopographicMaskMap(
            Texture2D sourceMasks,
            Texture2D existingMask,
            float cliffDetailStrength
        )
        {
            int w = sourceMasks.width;
            int h = sourceMasks.height;

            Texture2D target = existingMask;
            if (target != null && (target.width != w || target.height != h || target.format != TextureFormat.RGBA32))
            {
                DerivedMaskStampCache.Remove(target);
                DestroyObjectSafe(target);
                target = null;
            }

            if (target != null &&
                DerivedMaskStampCache.TryGetValue(target, out DerivedMaskStamp stamp) &&
                stamp.SourceMask == sourceMasks &&
                Mathf.Abs(stamp.CliffBoost - cliffDetailStrength) < 0.01f)
            {
                return target;
            }

            if (target == null)
            {
                target = new Texture2D(w, h, TextureFormat.RGBA32, mipChain: true, linear: true)
                {
                    name = DerivedTopographicMaskName,
                    wrapMode = TextureWrapMode.Clamp,
                    filterMode = FilterMode.Bilinear
                };
            }

            Color32[] srcPixels = sourceMasks.GetPixels32();
            var dstPixels = new Color32[srcPixels.Length];

            // Subtle AO floor keeps sunlit regions bright while accentuating crevices and steep canyon walls
            float aoWeight = Mathf.Clamp01(0.45f + cliffDetailStrength * 0.25f);

            for (int i = 0; i < srcPixels.Length; i++)
            {
                Color32 s = srcPixels[i];
                float slope = s.r * (1f / 255f);
                float rawAo = s.g * (1f / 255f);
                // Gently darken steep cliffs slightly in AO channel when cliff detail is boosted
                float cliffShade = 1f - slope * Mathf.Clamp01(cliffDetailStrength * 0.18f);
                float ao = Mathf.Lerp(1f, rawAo, aoWeight) * cliffShade;
                byte aoByte = (byte)Mathf.RoundToInt(Mathf.Clamp01(ao) * 255f);

                // Matte terrain smoothness (~0.08 flat, even rougher 0.03 on steep rock faces)
                float smoothness = Mathf.Lerp(0.08f, 0.03f, slope);
                byte smoothByte = (byte)Mathf.RoundToInt(smoothness * 255f);

                dstPixels[i] = new Color32(0, aoByte, s.b, smoothByte);
            }

            target.SetPixels32(dstPixels);
            target.Apply(updateMipmaps: true, makeNoLongerReadable: false);
            DerivedMaskStampCache.Remove(target);
            DerivedMaskStampCache.Add(target, new DerivedMaskStamp
            {
                SourceMask = sourceMasks,
                CliffBoost = cliffDetailStrength
            });
            return target;
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
                    DestroyObjectSafe(col);
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

        private static Texture2D CreateMatteTerrainMaskTexture(float smoothness = 0.08f)
        {
            if (_sharedMatteMask != null)
                return _sharedMatteMask;

            byte s = (byte)Mathf.RoundToInt(Mathf.Clamp01(smoothness) * 255f);
            var tex = new Texture2D(2, 2, TextureFormat.RGBA32, mipChain: false, linear: true)
            {
                name = "DepthPro2_MatteMaskMap",
                wrapMode = TextureWrapMode.Repeat
            };
            var c = new Color32(0, 255, 0, s);
            tex.SetPixels32(new[] { c, c, c, c });
            tex.Apply(false, false);
            _sharedMatteMask = tex;
            return tex;
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
