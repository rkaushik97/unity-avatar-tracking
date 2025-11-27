using UnityEngine;
using UnityEngine.AI;

public class AvatarWalker : MonoBehaviour
{
    [Header("Walking Settings")]
    public Transform[] waypoints;
    public float walkSpeed = 2.0f;
    public float stoppingDistance = 0.5f;
    
    [Header("Animation Parameters")]
    public Animator animator;
    public string speedParameter = "Speed";
    
    private int currentWaypointIndex = 0;
    private NavMeshAgent navMeshAgent;
    private bool isWalking = false;

    void Start()
    {
        // Get components
        animator = GetComponent<Animator>();
        navMeshAgent = GetComponent<NavMeshAgent>();
        
        // Setup NavMeshAgent
        if (navMeshAgent != null)
        {
            navMeshAgent.speed = walkSpeed;
            navMeshAgent.stoppingDistance = stoppingDistance;
        }
        
        // Start walking if waypoints are set
        if (waypoints != null && waypoints.Length > 0)
        {
            StartWalking();
        }
    }

    void Update()
    {
        if (isWalking)
        {
            // Update animation based on movement
            if (navMeshAgent != null)
            {
                float speed = navMeshAgent.velocity.magnitude;
                animator.SetFloat(speedParameter, speed);
                
                // Check if reached current waypoint
                if (!navMeshAgent.pathPending && navMeshAgent.remainingDistance <= stoppingDistance)
                {
                    GoToNextWaypoint();
                }
            }
            else
            {
                // Fallback without NavMeshAgent
                ManualMovement();
            }
        }
    }

    public void StartWalking()
    {
        if (waypoints == null || waypoints.Length == 0)
        {
            Debug.LogWarning("No waypoints set for walking!");
            return;
        }
        
        isWalking = true;
        currentWaypointIndex = 0;
        
        if (navMeshAgent != null)
        {
            navMeshAgent.SetDestination(waypoints[currentWaypointIndex].position);
        }
        
        animator.SetFloat(speedParameter, walkSpeed);
    }

    public void StopWalking()
    {
        isWalking = false;
        if (navMeshAgent != null)
        {
            navMeshAgent.isStopped = true;
        }
        animator.SetFloat(speedParameter, 0f);
    }

    private void GoToNextWaypoint()
    {
        currentWaypointIndex = (currentWaypointIndex + 1) % waypoints.Length;
        
        if (navMeshAgent != null)
        {
            navMeshAgent.SetDestination(waypoints[currentWaypointIndex].position);
        }
    }

    private void ManualMovement()
    {
        Transform target = waypoints[currentWaypointIndex];
        Vector3 direction = (target.position - transform.position).normalized;
        
        // Move towards waypoint
        transform.position = Vector3.MoveTowards(transform.position, target.position, walkSpeed * Time.deltaTime);
        
        // Rotate towards movement direction
        if (direction != Vector3.zero)
        {
            Quaternion targetRotation = Quaternion.LookRotation(direction);
            transform.rotation = Quaternion.Slerp(transform.rotation, targetRotation, Time.deltaTime * 5f);
        }
        
        // Check if reached waypoint
        if (Vector3.Distance(transform.position, target.position) <= stoppingDistance)
        {
            GoToNextWaypoint();
        }
    }

    // Draw waypoints in editor
    void OnDrawGizmos()
    {
        if (waypoints != null && waypoints.Length > 0)
        {
            Gizmos.color = Color.blue;
            for (int i = 0; i < waypoints.Length; i++)
            {
                if (waypoints[i] != null)
                {
                    Gizmos.DrawSphere(waypoints[i].position, 0.2f);
                    if (i < waypoints.Length - 1 && waypoints[i + 1] != null)
                    {
                        Gizmos.DrawLine(waypoints[i].position, waypoints[i + 1].position);
                    }
                }
            }
        }
    }
}