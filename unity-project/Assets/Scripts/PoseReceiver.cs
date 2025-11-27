using UnityEngine;
using UnityEngine.Networking;
using System.Collections;

public class PoseReceiver : MonoBehaviour
{
    // Flask server URL (adjust if running on another machine)
    public string serverUrl = "http://localhost:5000/pose";

    void Start()
    {
        // Start polling the server
        StartCoroutine(FetchPoseData());
    }

    IEnumerator FetchPoseData()
    {
        while (true)
        {
            using (UnityWebRequest www = UnityWebRequest.Get(serverUrl))
            {
                yield return www.SendWebRequest();

                if (www.result == UnityWebRequest.Result.Success)
                {
                    // Print the JSON response to the console
                    Debug.Log("Pose JSON: " + www.downloadHandler.text);

                    // TODO: You can parse this JSON and move your avatar
                    // For now, just printing is enough for testing
                }
                else
                {
                    Debug.LogWarning("Error fetching pose: " + www.error);
                }
            }

            // Fetch every 0.033s (~30 FPS)
            yield return new WaitForSeconds(0.033f);
        }
    }
}
