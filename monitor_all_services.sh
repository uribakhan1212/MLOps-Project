#!/bin/bash
# Monitor all MLOps services during pipeline execution

set -e

K8S_NAMESPACE="mlops-fl"

echo "🚀 Setting up monitoring for all MLOps services..."
echo ""

# Check if services are running
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Checking service status..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
kubectl get pods -n $K8S_NAMESPACE

echo ""
echo "Services and their ports:"
kubectl get svc -n $K8S_NAMESPACE

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Starting port-forwards..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Start Jenkins port-forward
echo "Starting Jenkins..."
kubectl port-forward -n $K8S_NAMESPACE svc/jenkins 8081:8080 > /dev/null 2>&1 &
PF_JENKINS=$!
echo "✓ Jenkins: http://localhost:8081"

# Start MLflow port-forward
echo "Starting MLflow..."
kubectl port-forward -n $K8S_NAMESPACE svc/mlflow 8082:80 > /dev/null 2>&1 &
PF_MLFLOW=$!
echo "✓ MLflow: http://localhost:8082"

# Start Prometheus port-forward
echo "Starting Prometheus..."
kubectl port-forward -n $K8S_NAMESPACE svc/prometheus-server 9090:80 > /dev/null 2>&1 &
PF_PROMETHEUS=$!
echo "✓ Prometheus: http://localhost:9090"

# Start Grafana port-forward
echo "Starting Grafana..."
kubectl port-forward -n $K8S_NAMESPACE svc/grafana 3000:80 > /dev/null 2>&1 &
PF_GRAFANA=$!
echo "✓ Grafana: http://localhost:3000"

# Start Inference API port-forward
echo "Starting Inference API..."
# Try to detect the correct service and port
INFERENCE_SVC=$(kubectl get svc -n $K8S_NAMESPACE -o name | grep -i inference | head -1 | cut -d'/' -f2)
if [ -z "$INFERENCE_SVC" ]; then
    INFERENCE_SVC="diabetes-inference-service"
fi

# Use the correct service port (80 -> container port 5003)
if kubectl port-forward -n $K8S_NAMESPACE svc/$INFERENCE_SVC 5003:80 > /dev/null 2>&1 &
then
    PF_API=$!
    echo "✓ Inference API: http://localhost:5003 (service port 80)"
else
    echo "⚠️ Could not start Inference API port-forward"
    echo "   Service: $INFERENCE_SVC"
    echo "   Check: kubectl get svc -n $K8S_NAMESPACE | grep inference"
    PF_API=""
fi

sleep 5

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ All port-forwards active!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "📊 Access your services:"
echo ""
echo "  Jenkins:        http://localhost:8081"
echo "  MLflow:         http://localhost:8082"
echo "  Prometheus:     http://localhost:9090"
echo "  Grafana:        http://localhost:3000"
echo "  Inference API:  http://localhost:5003"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "💡 Monitoring Tips:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "1. Jenkins (Pipeline Execution):"
echo "   - Watch build progress in real-time"
echo "   - View console output for each stage"
echo "   - Monitor stage transitions"
echo ""
echo "2. MLflow (Training Metrics):"
echo "   - Go to: Experiments → diabetes-federated-learning"
echo "   - Watch runs appear during training"
echo "   - See metrics update per round"
echo "   - Check Model Registry for new versions"
echo ""
echo "3. Prometheus (Metrics Collection):"
echo "   - Go to: Status → Targets"
echo "   - Verify all targets are UP"
echo "   - Run queries: flask_http_request_total"
echo "   - Check: Graph → Execute query"
echo ""
echo "4. Grafana (Visual Dashboards):"
echo "   - View real-time request metrics"
echo "   - Monitor pod health"
echo "   - Track response times"
echo "   - Set up alerts"
echo ""
echo "5. Inference API (Direct Testing):"
echo "   - Health: curl http://localhost:5003/health"
echo "   - Metrics: curl http://localhost:5003/metrics"
echo "   - Predict: curl -X POST http://localhost:5003/predict -d '{...}'"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🎯 Quick Commands:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "# Test API"
echo "curl -X POST http://localhost:5003/predict \\"
echo "  -H 'Content-Type: application/json' \\"
echo "  -d '{\"HighBP\":1,\"HighChol\":1,\"BMI\":28.5,...}'"
echo ""
echo "# Watch Kubernetes pods"
echo "watch kubectl get pods -n $K8S_NAMESPACE"
echo ""
echo "# Stream Jenkins logs"
echo "kubectl logs -f -n $K8S_NAMESPACE -l app.kubernetes.io/name=jenkins"
echo ""
echo "# Check Prometheus targets"
echo "curl -s http://localhost:9090/api/v1/targets | jq '.data.activeTargets[] | {job: .labels.job, health: .health}'"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "📈 Open these URLs in your browser now!"
echo ""
echo "Press Ctrl+C to stop all port-forwards"
echo ""

# Cleanup on exit
cleanup() {
    echo ""
    echo "Stopping port-forwards..."
    for pid in $PF_JENKINS $PF_MLFLOW $PF_PROMETHEUS $PF_GRAFANA $PF_API; do
        if [ ! -z "$pid" ]; then
            kill $pid 2>/dev/null || true
        fi
    done
    echo "Port-forwards stopped"
}
trap cleanup EXIT

# Keep script running
wait