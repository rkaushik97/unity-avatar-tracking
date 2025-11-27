using UnityEngine;
using System.Collections;
using System.Collections.Generic;
using UnityEngine.Networking;
using System;

public class StreamToBackend : MonoBehaviour
{
    [Header("Stream Settings")]
    public Camera targetCamera;
    public string backendURL = "http://localhost:5000/stream";
    public int targetFPS = 30;
    public int imageWidth = 1280;
    public int imageHeight = 720;
    
    [Header("Status")]
    public bool isStreaming = false;
    public int framesSent = 0;
    public float currentFPS = 0f;
    
    private RenderTexture renderTexture;
    private Texture2D screenShot;
    private float nextFrameTime = 0f;
    private float fpsTimer = 0f;
    private int fpsCounter = 0;

    void Start()
    {
        // Use main camera if none assigned
        if (targetCamera == null)
        {
            targetCamera = Camera.main;
        }

        // Create render texture
        renderTexture = new RenderTexture(imageWidth, imageHeight, 24);
        screenShot = new Texture2D(imageWidth, imageHeight, TextureFormat.RGB24, false);
        
        Debug.Log("Stream initialized. Press SPACE to start/stop streaming.");
    }

    void Update()
    {
        // Toggle streaming with spacebar
        if (Input.GetKeyDown(KeyCode.Space))
        {
            isStreaming = !isStreaming;
            Debug.Log("Streaming: " + (isStreaming ? "ON" : "OFF"));
        }

        // Calculate FPS
        fpsTimer += Time.deltaTime;
        if (fpsTimer >= 1f)
        {
            currentFPS = fpsCounter;
            fpsCounter = 0;
            fpsTimer = 0f;
        }

        // Stream frames
        if (isStreaming && Time.time >= nextFrameTime)
        {
            nextFrameTime = Time.time + (1f / targetFPS);
            StartCoroutine(CaptureAndSendFrame());
            fpsCounter++;
        }
    }

    IEnumerator CaptureAndSendFrame()
    {
        // Capture frame from camera
        targetCamera.targetTexture = renderTexture;
        targetCamera.Render();
        
        RenderTexture.active = renderTexture;
        screenShot.ReadPixels(new Rect(0, 0, imageWidth, imageHeight), 0, 0);
        screenShot.Apply();
        
        targetCamera.targetTexture = null;
        RenderTexture.active = null;

        // Encode to JPG
        byte[] bytes = screenShot.EncodeToJPG(75);
        string base64Frame = Convert.ToBase64String(bytes);

        // Create JSON payload
        string json = "{\"frame\":\"" + base64Frame + "\"}";

        // Send to backend
        UnityWebRequest request = new UnityWebRequest(backendURL, "POST");
        byte[] bodyRaw = System.Text.Encoding.UTF8.GetBytes(json);
        request.uploadHandler = new UploadHandlerRaw(bodyRaw);
        request.downloadHandler = new DownloadHandlerBuffer();
        request.SetRequestHeader("Content-Type", "application/json");

        yield return request.SendWebRequest();

        if (request.result == UnityWebRequest.Result.Success)
        {
            framesSent++;
        }
        else
        {
            Debug.LogWarning("Failed to send frame: " + request.error);
        }

        request.Dispose();
    }

    void OnGUI()
    {
        GUI.Box(new Rect(10, 10, 300, 100), "Unity Stream Status");
        GUI.Label(new Rect(20, 35, 280, 20), "Streaming: " + (isStreaming ? "ON" : "OFF"));
        GUI.Label(new Rect(20, 55, 280, 20), "Frames Sent: " + framesSent);
        GUI.Label(new Rect(20, 75, 280, 20), "FPS: " + currentFPS.ToString("F1"));
        GUI.Label(new Rect(20, 95, 280, 20), "Press SPACE to toggle");
    }

    void OnDestroy()
    {
        if (renderTexture != null)
        {
            renderTexture.Release();
            Destroy(renderTexture);
        }
        if (screenShot != null)
        {
            Destroy(screenShot);
        }
    }
}
