from fastapi.testclient import TestClient

from app.main import create_app


def test_health_endpoint_reports_service_ready():
    client = TestClient(create_app())

    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json() == {"service": "talent-document-api", "status": "ok"}


def test_openapi_exposes_lesson_5_chunking_workflow_without_removed_annotation_endpoints():
    client = TestClient(create_app())

    paths = client.get("/openapi.json").json()["paths"]

    assert "/api/documents/{document_id}/chunks" in paths
    assert set(paths["/api/documents/{document_id}/chunks"]) == {"get", "post"}
    assert "/api/documents/{document_id}/chunk-annotations" not in paths
    assert "/api/documents/{document_id}/chunk-evaluation" not in paths


def test_openapi_exposes_evidence_index_and_search_workflow():
    client = TestClient(create_app())

    paths = client.get("/openapi.json").json()["paths"]

    assert "/api/documents/{document_id}/evidence-index" in paths
    assert set(paths["/api/documents/{document_id}/evidence-index"]) == {"post"}
    assert set(paths["/api/documents"]) == {"get", "post", "delete"}
    assert set(paths["/api/knowledge-bases"]) == {"get", "post"}
    assert set(paths["/api/knowledge-bases/{knowledge_base_id}"]) == {"delete"}
    assert "/api/evidence/search" in paths
    assert set(paths["/api/evidence/search"]) == {"post"}
    assert "/api/index-jobs/{job_id}/retry" in paths
