TECH_GROUPS = {
    # ── BACKEND FRAMEWORKS ──────────────────────────────
    "spring boot":       "java_backend_framework",
    "spring":            "java_backend_framework",
    "spring framework":  "java_backend_framework",
    "quarkus":           "java_backend_framework",
    "micronaut":         "java_backend_framework",

    "django":            "python_backend_framework",
    "fastapi":           "python_backend_framework",
    "flask":             "python_backend_framework",

    "express":           "nodejs_backend_framework",
    "nestjs":            "nodejs_backend_framework",
    "node.js":           "nodejs_backend_framework",
    "node":              "nodejs_backend_framework",

    "asp.net":           "dotnet_backend_framework",
    "asp.net core":      "dotnet_backend_framework",
    ".net":              "dotnet_backend_framework",
    ".net core":         "dotnet_backend_framework",

    "laravel":           "php_backend_framework",
    "symfony":           "php_backend_framework",
    "codeigniter":       "php_backend_framework",

    "ruby on rails":     "ruby_backend_framework",
    "rails":             "ruby_backend_framework",

    # ── FRONTEND FRAMEWORKS ─────────────────────────────
    "react":             "js_frontend_framework",
    "react.js":          "js_frontend_framework",
    "reactjs":           "js_frontend_framework",
    "vue":               "js_frontend_framework",
    "vue.js":            "js_frontend_framework",
    "vuejs":             "js_frontend_framework",
    "angular":           "js_frontend_framework",
    "angularjs":         "js_frontend_framework",
    "svelte":            "js_frontend_framework",
    "solid.js":          "js_frontend_framework",

    "next.js":           "js_ssr_framework",
    "nextjs":            "js_ssr_framework",
    "nuxt.js":           "js_ssr_framework",
    "nuxtjs":            "js_ssr_framework",
    "remix":             "js_ssr_framework",

    # ── CSS / UI FRAMEWORKS ─────────────────────────────
    "tailwind":          "css_framework",
    "tailwind css":      "css_framework",
    "bootstrap":         "css_framework",
    "bulma":             "css_framework",
    "material ui":       "css_framework",
    "mui":               "css_framework",
    "shadcn":            "css_framework",
    "shadcn/ui":         "css_framework",
    "ant design":        "css_framework",
    "chakra ui":         "css_framework",

    # ── MOBILE FRAMEWORKS ───────────────────────────────
    "flutter":           "cross_platform_mobile",
    "react native":      "cross_platform_mobile",
    "kotlin multiplatform": "cross_platform_mobile",
    "xamarin":           "cross_platform_mobile",
    "ionic":             "cross_platform_mobile",

    "kotlin":            "native_android",
    "java android":      "native_android",

    "swift":             "native_ios",
    "swiftui":           "native_ios",

    # ── RELATIONAL DATABASES ────────────────────────────
    "postgresql":        "rdbms",
    "postgres":          "rdbms",
    "mysql":             "rdbms",
    "mariadb":           "rdbms",
    "mssql":             "rdbms",
    "sql server":        "rdbms",
    "sqlite":            "rdbms",
    "oracle":            "rdbms",

    # ── NOSQL — DOCUMENT ────────────────────────────────
    "mongodb":           "document_db",
    "couchdb":           "document_db",
    "firestore":         "document_db",
    "firebase firestore":"document_db",
    "dynamodb":          "document_db",
    "cosmosdb":          "document_db",

    # ── NOSQL — KEY-VALUE / CACHE ───────────────────────
    "redis":             "cache_store",
    "memcached":         "cache_store",

    # ── NOSQL — SEARCH ──────────────────────────────────
    "elasticsearch":     "search_engine",
    "opensearch":        "search_engine",
    "solr":              "search_engine",

    # ── CONTAINERIZATION ────────────────────────────────
    "docker":            "container_runtime",
    "podman":            "container_runtime",
    "containerd":        "container_runtime",

    # ── CONTAINER ORCHESTRATION ─────────────────────────
    "kubernetes":        "container_orchestration",
    "k8s":               "container_orchestration",
    "docker swarm":      "container_orchestration",

    # ── CI/CD ───────────────────────────────────────────
    "github actions":    "cicd_tool",
    "gitlab ci":         "cicd_tool",
    "jenkins":           "cicd_tool",
    "circle ci":         "cicd_tool",
    "travis ci":         "cicd_tool",

    # ── CLOUD PLATFORMS ─────────────────────────────────
    "aws":               "cloud_platform",
    "amazon web services": "cloud_platform",
    "azure":             "cloud_platform",
    "microsoft azure":   "cloud_platform",
    "gcp":               "cloud_platform",
    "google cloud":      "cloud_platform",

    # ── OBJECT STORAGE ──────────────────────────────────
    "aws s3":            "object_storage",
    "s3":                "object_storage",
    "azure blob":        "object_storage",
    "azure blob storage":"object_storage",
    "minio":             "object_storage",
    "gcs":               "object_storage",

    # ── MESSAGE QUEUE / BACKGROUND JOBS ─────────────────
    "rabbitmq":          "message_broker",
    "kafka":             "message_broker",
    "activemq":          "message_broker",
    "nats":              "message_broker",

    "celery":            "job_queue",
    "hangfire":          "job_queue",
    "bullmq":            "job_queue",
    "bull":              "job_queue",
    "rq":                "job_queue",
    "sidekiq":           "job_queue",

    # ── BAAS / BACKEND AS A SERVICE ─────────────────────
    "firebase":          "baas",
    "supabase":          "baas",
    "pocketbase":        "baas",
    "appwrite":          "baas",

    # ── API STYLES ──────────────────────────────────────
    "rest":              "api_style",
    "rest api":          "api_style",
    "restful":           "api_style",
    "restful api":       "api_style",
    "graphql":           "api_style",
    "grpc":              "api_style",

    # ── AUTH ────────────────────────────────────────────
    "jwt":               "auth_mechanism",
    "oauth":             "auth_mechanism",
    "oauth2":            "auth_mechanism",
    "openid connect":    "auth_mechanism",
    "keycloak":          "auth_mechanism",
    "auth0":             "auth_mechanism",

    # ── AI / ML FRAMEWORKS ──────────────────────────────
    "tensorflow":        "ml_framework",
    "pytorch":           "ml_framework",
    "keras":             "ml_framework",
    "scikit-learn":      "ml_framework",
    "sklearn":           "ml_framework",
    "xgboost":           "ml_framework",

    "langchain":         "llm_framework",
    "llamaindex":        "llm_framework",
    "llama index":       "llm_framework",
    "haystack":          "llm_framework",

    "openai":            "llm_provider",
    "gpt":               "llm_provider",
    "gpt-4":             "llm_provider",
    "gpt-3.5":           "llm_provider",
    "claude":            "llm_provider",
    "gemini":            "llm_provider",
    "llama":             "llm_provider",
    "ollama":            "llm_provider",

    # ── VECTOR DATABASE ─────────────────────────────────
    "pinecone":          "vector_db",
    "weaviate":          "vector_db",
    "chroma":            "vector_db",
    "chromadb":          "vector_db",
    "qdrant":            "vector_db",
    "faiss":             "vector_db",
    "milvus":            "vector_db",

    # ── MONITORING / OBSERVABILITY ──────────────────────
    "prometheus":        "monitoring_tool",
    "grafana":           "monitoring_tool",
    "datadog":           "monitoring_tool",
    "sentry":            "monitoring_tool",

    # ── REVERSE PROXY / LOAD BALANCER ───────────────────
    "nginx":             "reverse_proxy",
    "apache":            "reverse_proxy",
    "traefik":           "reverse_proxy",
    "caddy":             "reverse_proxy",

    # ── PAYMENT ─────────────────────────────────────────
    "stripe":            "payment_gateway",
    "paypal":            "payment_gateway",
    "vnpay":             "payment_gateway",
    "momo":              "payment_gateway",
    "zalopay":           "payment_gateway",
    "payos":             "payment_gateway",
}


def normalize_tech(keyword: str) -> str:
    """
    Normalize keyword về canonical group.
    Return group name nếu tìm thấy, ngược lại return keyword gốc.
    """
    return TECH_GROUPS.get(keyword.lower().strip(), keyword.lower().strip())


def tech_dist_content(label_u: str, label_v: str) -> float:
    """
    Tính dist_content có tech-awareness.
    
    1. Nếu cùng tech group → dist = 0.0 (tương đương)
    2. Nếu khác group → Jaccard trên tokens
    """
    group_u = normalize_tech(label_u)
    group_v = normalize_tech(label_v)

    # Cùng group → tương đương hoàn toàn
    if group_u == group_v:
        return 0.0

    # Khác group → Jaccard trên normalized labels
    tok_u = set(group_u.split("_"))
    tok_v = set(group_v.split("_"))
    union = tok_u | tok_v
    if not union:
        return 1.0
    jaccard_sim = len(tok_u & tok_v) / len(union)
    return 1.0 - jaccard_sim