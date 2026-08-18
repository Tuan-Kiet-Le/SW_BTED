import os

def generate_report():
    html_content = """<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Báo Cáo Phân Tích Lỗi Thực Nghiệm (Error Analysis Report)</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-color: #0b0f19;
            --card-bg: rgba(22, 28, 45, 0.6);
            --border-color: rgba(255, 255, 255, 0.08);
            --text-primary: #f3f4f6;
            --text-secondary: #9ca3af;
            --accent-blue: #3b82f6;
            --accent-green: #10b981;
            --accent-red: #ef4444;
            --accent-purple: #8b5cf6;
            --accent-amber: #f59e0b;
            --sidebar-bg: #111827;
            --tag-fn-bg: rgba(239, 68, 68, 0.12);
            --tag-fn-color: #ef4444;
            --tag-fp-bg: rgba(245, 158, 11, 0.12);
            --tag-fp-color: #f59e0b;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            font-family: 'Inter', sans-serif;
            background-color: var(--bg-color);
            color: var(--text-primary);
            display: flex;
            min-height: 100vh;
            overflow-x: hidden;
        }

        /* Sidebar Navigation */
        .sidebar {
            width: 320px;
            background-color: var(--sidebar-bg);
            border-right: 1px solid var(--border-color);
            padding: 2.5rem 1.5rem;
            display: flex;
            flex-direction: column;
            gap: 1.5rem;
            flex-shrink: 0;
        }

        .sidebar-title {
            font-size: 1.25rem;
            font-weight: 700;
            color: #fff;
            letter-spacing: -0.5px;
            margin-bottom: 1rem;
            background: linear-gradient(135deg, #a78bfa, #8b5cf6);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .nav-btn {
            background: transparent;
            border: 1px solid transparent;
            color: var(--text-secondary);
            padding: 1.2rem;
            border-radius: 12px;
            text-align: left;
            font-size: 0.95rem;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            display: flex;
            flex-direction: column;
            gap: 0.3rem;
        }

        .nav-btn:hover {
            background: rgba(255, 255, 255, 0.03);
            color: #fff;
            border-color: rgba(255, 255, 255, 0.05);
        }

        .nav-btn.active {
            background: rgba(139, 92, 246, 0.15);
            border-color: rgba(139, 92, 246, 0.4);
            color: #c084fc;
            font-weight: 600;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
        }

        .nav-btn span {
            font-size: 0.75rem;
            color: rgba(255, 255, 255, 0.4);
            font-family: monospace;
            text-transform: uppercase;
        }

        /* Main Content */
        .main-content {
            flex-grow: 1;
            padding: 3rem;
            overflow-y: auto;
            max-height: 100vh;
        }

        .header {
            margin-bottom: 2.5rem;
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 1.5rem;
        }

        .header h1 {
            font-size: 2.2rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
            letter-spacing: -0.75px;
            background: linear-gradient(135deg, #fff, #9ca3af);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .header p {
            color: var(--text-secondary);
            font-size: 1.05rem;
        }

        /* Report View Section */
        .report-section {
            display: none;
            flex-direction: column;
            gap: 2rem;
        }

        .report-section.active {
            display: flex;
        }

        /* Case Card */
        .case-card {
            background: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            overflow: hidden;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.15);
            transition: transform 0.2s ease, border-color 0.2s ease;
        }

        .case-card:hover {
            border-color: rgba(255, 255, 255, 0.15);
        }

        .case-header {
            background: rgba(255, 255, 255, 0.02);
            padding: 1.2rem 1.8rem;
            border-bottom: 1px solid var(--border-color);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .case-title-area {
            display: flex;
            align-items: center;
            gap: 0.8rem;
        }

        .case-code {
            font-family: monospace;
            font-size: 1rem;
            font-weight: 700;
            color: #fff;
            background: rgba(255, 255, 255, 0.06);
            padding: 0.3rem 0.6rem;
            border-radius: 6px;
        }

        .case-type-badge {
            font-size: 0.75rem;
            font-weight: 600;
            padding: 0.3rem 0.6rem;
            border-radius: 6px;
            background: rgba(59, 130, 246, 0.12);
            color: #60a5fa;
            text-transform: uppercase;
        }

        .case-error-tag {
            font-size: 0.75rem;
            font-weight: 700;
            padding: 0.4rem 0.8rem;
            border-radius: 20px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        .tag-fn {
            background: var(--tag-fn-bg);
            color: var(--tag-fn-color);
            border: 1px solid rgba(239, 68, 68, 0.2);
        }

        .tag-fp {
            background: var(--tag-fp-bg);
            color: var(--tag-fp-color);
            border: 1px solid rgba(245, 158, 11, 0.2);
        }

        /* Grid info */
        .case-body {
            padding: 1.8rem;
        }

        .docs-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1.5rem;
            margin-bottom: 1.5rem;
        }

        .doc-info-box {
            background: rgba(0, 0, 0, 0.2);
            border: 1px solid rgba(255, 255, 255, 0.03);
            border-radius: 12px;
            padding: 1.2rem;
        }

        .doc-label {
            font-size: 0.75rem;
            color: var(--text-secondary);
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 0.5rem;
        }

        .doc-title {
            font-size: 1.05rem;
            font-weight: 600;
            color: #fff;
            line-height: 1.5;
        }

        /* Metric Bar */
        .metrics-row {
            display: flex;
            gap: 1.5rem;
            margin-bottom: 1.5rem;
            border-top: 1px dashed var(--border-color);
            border-bottom: 1px dashed var(--border-color);
            padding: 1rem 0;
        }

        .metric-card {
            flex: 1;
            display: flex;
            flex-direction: column;
            gap: 0.25rem;
        }

        .metric-name {
            font-size: 0.8rem;
            color: var(--text-secondary);
            font-weight: 500;
        }

        .metric-value-box {
            display: flex;
            align-items: baseline;
            gap: 0.5rem;
        }

        .metric-value {
            font-size: 1.25rem;
            font-weight: 700;
            font-family: monospace;
        }

        .metric-status {
            font-size: 0.75rem;
            font-weight: 600;
        }

        .status-crossed {
            color: var(--accent-red);
        }

        .status-safe {
            color: var(--accent-green);
        }

        /* Explanation area */
        .explanation-box {
            background: rgba(139, 92, 246, 0.03);
            border-left: 4px solid var(--accent-purple);
            padding: 1.2rem 1.5rem;
            border-radius: 0 12px 12px 0;
            line-height: 1.6;
        }

        .explanation-box h4 {
            font-size: 0.95rem;
            font-weight: 700;
            color: #c084fc;
            margin-bottom: 0.4rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .explanation-text {
            font-size: 0.95rem;
            color: var(--text-primary);
        }

        .empty-state {
            text-align: center;
            padding: 4rem 2rem;
            color: var(--text-secondary);
            font-size: 1.1rem;
            background: var(--card-bg);
            border: 1px dashed var(--border-color);
            border-radius: 16px;
        }
    </style>
</head>
<body>

    <!-- Sidebar Navigation -->
    <div class="sidebar">
        <div class="sidebar-title">Mô hình Đánh giá</div>
        <button class="nav-btn active" onclick="switchTab('sw-bted', this)">
            <span>Proposed Model</span>
            SW-BTED
            <p style="font-size:0.75rem; color:var(--accent-purple); margin-top:0.2rem;">5 lỗi / 180 cặp</p>
        </button>
        <button class="nav-btn" onclick="switchTab('b1-tfidf', this)">
            <span>Baseline B1</span>
            Cosine TF-IDF
            <p style="font-size:0.75rem; color:var(--accent-amber); margin-top:0.2rem;">1 lỗi / 180 cặp</p>
        </button>
        <button class="nav-btn" onclick="switchTab('b2-sbert', this)">
            <span>Baseline B2</span>
            Cosine SBERT
            <p style="font-size:0.75rem; color:var(--accent-amber); margin-top:0.2rem;">1 lỗi / 180 cặp</p>
        </button>
    </div>

    <!-- Main Content -->
    <div class="main-content">
        <div class="header">
            <h1>Thống Kê Cặp Lỗi Thực Nghiệm (Error Analysis)</h1>
            <p>Trực quan hóa và phân tích khoa học chi tiết các trường hợp phân loại sai của thuật toán đề xuất và các baseline.</p>
        </div>

        <!-- TAB SW-BTED -->
        <div class="report-section active" id="tab-sw-bted">
            <!-- Case 1 -->
            <div class="case-card">
                <div class="case-header">
                    <div class="case-title-area">
                        <div class="case-code">Cặp #11</div>
                        <span class="case-type-badge">Type A (Plagiarism)</span>
                    </div>
                    <span class="case-error-tag tag-fn">False Negative (Bỏ lọt)</span>
                </div>
                <div class="case-body">
                    <div class="docs-grid">
                        <div class="doc-info-box">
                            <div class="doc-label">📄 Tài liệu A (Gốc)</div>
                            <div class="doc-title">SU26SE048: On-demand Home Care Service Application</div>
                        </div>
                        <div class="doc-info-box">
                            <div class="doc-label">✨ Tài liệu B (Paraphrase)</div>
                            <div class="doc-title">SU26SE048_plag: On-demand Home Care Service Application (Variant)</div>
                        </div>
                    </div>
                    <div class="metrics-row">
                        <div class="metrics-row" style="border:none; padding:0; margin:0; width:100%; display:flex; gap:1.5rem;">
                            <div class="metric-card">
                                <div class="metric-name">Độ tương đồng SW-BTED</div>
                                <div class="metric-value-box">
                                    <span class="metric-value" style="color: var(--accent-red)">0.3945</span>
                                    <span class="metric-status status-crossed">Dưới ngưỡng (0.40)</span>
                                </div>
                            </div>
                            <div class="metric-card">
                                <div class="metric-name">TF-IDF Similarity</div>
                                <div class="metric-value-box">
                                    <span class="metric-value">0.5915</span>
                                </div>
                            </div>
                            <div class="metric-card">
                                <div class="metric-name">SBERT Similarity</div>
                                <div class="metric-value-box">
                                    <span class="metric-value">0.8769</span>
                                </div>
                            </div>
                        </div>
                    </div>
                    <div class="explanation-box">
                        <h4>🔍 Phân tích khoa học</h4>
                        <div class="explanation-text">
                            <strong>Trường hợp biên (Borderline case):</strong> Điểm tương đồng SW-BTED đạt <strong>0.3945</strong>, chỉ thấp hơn ngưỡng tối ưu (0.40) đúng 0.0055. Việc paraphrase của GPT-4o-mini diễn đạt lại cực kỳ mượt mà, phá vỡ một số liên kết từ khóa RAKE và làm thay đổi nhẹ cấu trúc phân tầng của các nút nhánh, khiến điểm số tổng thể bị tụt nhẹ dưới ngưỡng quyết định.
                        </div>
                    </div>
                </div>
            </div>

            <!-- Case 2 -->
            <div class="case-card">
                <div class="case-header">
                    <div class="case-title-area">
                        <div class="case-code">Cặp #41</div>
                        <span class="case-type-badge">Type A (Plagiarism)</span>
                    </div>
                    <span class="case-error-tag tag-fn">False Negative (Bỏ lọt)</span>
                </div>
                <div class="case-body">
                    <div class="docs-grid">
                        <div class="doc-info-box">
                            <div class="doc-label">📄 Tài liệu A (Gốc)</div>
                            <div class="doc-title">SP26SE140: Automatic Grading System for Java OOP Practical Exams</div>
                        </div>
                        <div class="doc-info-box">
                            <div class="doc-label">✨ Tài liệu B (Paraphrase)</div>
                            <div class="doc-title">SP26SE140_plag: Automatic Grading System for Java OOP Practical Exams (Variant)</div>
                        </div>
                    </div>
                    <div class="metrics-row">
                        <div class="metrics-row" style="border:none; padding:0; margin:0; width:100%; display:flex; gap:1.5rem;">
                            <div class="metric-card">
                                <div class="metric-name">Độ tương đồng SW-BTED</div>
                                <div class="metric-value-box">
                                    <span class="metric-value" style="color: var(--accent-red)">0.3699</span>
                                    <span class="metric-status status-crossed">Dưới ngưỡng (0.40)</span>
                                </div>
                            </div>
                            <div class="metric-card">
                                <div class="metric-name">TF-IDF Similarity</div>
                                <div class="metric-value-box">
                                    <span class="metric-value">0.6163</span>
                                </div>
                            </div>
                            <div class="metric-card">
                                <div class="metric-name">SBERT Similarity</div>
                                <div class="metric-value-box">
                                    <span class="metric-value">0.9072</span>
                                </div>
                            </div>
                        </div>
                    </div>
                    <div class="explanation-box">
                        <h4>🔍 Phân tích khoa học</h4>
                        <div class="explanation-text">
                            <strong>Paraphrase triệt để:</strong> Bản paraphrase của GPT-4o-mini đối với phần giải pháp chuyên môn đã thay thế hầu hết các từ khóa lập trình đặc trưng (như Java, OOP, Exams, Grading) bằng các cụm mô tả tương đương khác. Do cây của SW-BTED trích xuất từ khóa thô qua RAKE và so khớp nghiêm ngặt theo mô hình chi phí ontology, sự thay đổi toàn diện từ khóa đã kéo điểm SW-BTED xuống 0.3699. Trái lại, các flat-text baseline vẫn nhận diện được do cấu trúc câu và bối cảnh tổng thể còn lưu giữ lượng từ vựng trùng lặp nhỏ.
                        </div>
                    </div>
                </div>
            </div>

            <!-- Case 3 -->
            <div class="case-card">
                <div class="case-header">
                    <div class="case-title-area">
                        <div class="case-code">Cặp #60</div>
                        <span class="case-type-badge">Type A (Plagiarism)</span>
                    </div>
                    <span class="case-error-tag tag-fn">False Negative (Bỏ lọt)</span>
                </div>
                <div class="case-body">
                    <div class="docs-grid">
                        <div class="doc-info-box">
                            <div class="doc-label">📄 Tài liệu A (Gốc)</div>
                            <div class="doc-title">SU26SE170: Rancour - 3D Action Role-playing Game</div>
                        </div>
                        <div class="doc-info-box">
                            <div class="doc-label">✨ Tài liệu B (Paraphrase)</div>
                            <div class="doc-title">SU26SE170_plag: Rancour - 3D Action Role-playing Game (Variant)</div>
                        </div>
                    </div>
                    <div class="metrics-row">
                        <div class="metrics-row" style="border:none; padding:0; margin:0; width:100%; display:flex; gap:1.5rem;">
                            <div class="metric-card">
                                <div class="metric-name">Độ tương đồng SW-BTED</div>
                                <div class="metric-value-box">
                                    <span class="metric-value" style="color: var(--accent-red)">0.3994</span>
                                    <span class="metric-status status-crossed">Dưới ngưỡng (0.40)</span>
                                </div>
                            </div>
                            <div class="metric-card">
                                <div class="metric-name">TF-IDF Similarity</div>
                                <div class="metric-value-box">
                                    <span class="metric-value">0.6735</span>
                                </div>
                            </div>
                            <div class="metric-card">
                                <div class="metric-name">SBERT Similarity</div>
                                <div class="metric-value-box">
                                    <span class="metric-value">0.8528</span>
                                </div>
                            </div>
                        </div>
                    </div>
                    <div class="explanation-box">
                        <h4>🔍 Phân tích khoa học</h4>
                        <div class="explanation-text">
                            <strong>Sai lệch cực kỳ sát nút:</strong> Điểm số đạt <strong>0.3994</strong>, chỉ thiếu đúng <strong>0.0006</strong> để được phân loại là đạo văn. Phép biến đổi cấu trúc câu làm RAKE trích xuất thiếu một số từ khóa nhỏ ở các chương phụ, tạo ra sự sai lệch cực nhỏ về tổng chi phí hiệu chỉnh cây và làm tụt điểm sát sạt dưới ngưỡng.
                        </div>
                    </div>
                </div>
            </div>

            <!-- Case 4 -->
            <div class="case-card">
                <div class="case-header">
                    <div class="case-title-area">
                        <div class="case-code">Cặp #93</div>
                        <span class="case-type-badge">Type B (Same Domain)</span>
                    </div>
                    <span class="case-error-tag tag-fp">False Positive (Báo động giả)</span>
                </div>
                <div class="case-body">
                    <div class="docs-grid">
                        <div class="doc-info-box">
                            <div class="doc-label">📄 Tài liệu A (Healthcare)</div>
                            <div class="doc-title">SU26SE048: On-demand Home Care Service Application</div>
                        </div>
                        <div class="doc-info-box">
                            <div class="doc-label">📄 Tài liệu B (Healthcare)</div>
                            <div class="doc-title">SU26SE087: PetGuardian – Pet Care Monitoring and Support Platform</div>
                        </div>
                    </div>
                    <div class="metrics-row">
                        <div class="metrics-row" style="border:none; padding:0; margin:0; width:100%; display:flex; gap:1.5rem;">
                            <div class="metric-card">
                                <div class="metric-name">Độ tương đồng SW-BTED</div>
                                <div class="metric-value-box">
                                    <span class="metric-value" style="color: var(--accent-red)">0.4001</span>
                                    <span class="metric-status status-crossed">Vượt ngưỡng (0.40)</span>
                                </div>
                            </div>
                            <div class="metric-card">
                                <div class="metric-name">TF-IDF Similarity</div>
                                <div class="metric-value-box">
                                    <span class="metric-value">0.2394</span>
                                </div>
                            </div>
                            <div class="metric-card">
                                <div class="metric-name">SBERT Similarity</div>
                                <div class="metric-value-box">
                                    <span class="metric-value">0.5087</span>
                                </div>
                            </div>
                        </div>
                    </div>
                    <div class="explanation-box">
                        <h4>🔍 Phân tích khoa học</h4>
                        <div class="explanation-text">
                            <strong>Trùng lặp nghiệp vụ mô hình kinh doanh:</strong> Cả hai hệ thống đều thiết kế theo dạng dịch vụ chăm sóc theo yêu cầu (Uber-for-Care). Việc trùng lặp quy trình nghiệp vụ (đặt lịch, thanh toán, quản lý tài khoản, lịch trình) cùng các cấu trúc phân cấp tương đương đã đẩy điểm SW-BTED lên 0.4001, vượt nhẹ qua ngưỡng lọc 0.40. Cặp này cũng đánh lừa được cả Baseline B1 TF-IDF (đạt 0.2394 so với ngưỡng 0.24).
                        </div>
                    </div>
                </div>
            </div>

            <!-- Case 5 -->
            <div class="case-card">
                <div class="case-header">
                    <div class="case-title-area">
                        <div class="case-code">Cặp #117</div>
                        <span class="case-type-badge">Type B (Same Domain)</span>
                    </div>
                    <span class="case-error-tag tag-fp">False Positive (Báo động giả)</span>
                </div>
                <div class="case-body">
                    <div class="docs-grid">
                        <div class="doc-info-box">
                            <div class="doc-label">📄 Tài liệu A (Healthcare - Random fallback)</div>
                            <div class="doc-title">SU26SE029: AI-Powered Drone Inspection System for Thermal Anomaly Detection on Solar Panels</div>
                        </div>
                        <div class="doc-info-box">
                            <div class="doc-label">📄 Tài liệu B (Healthcare)</div>
                            <div class="doc-title">SU26SE087: PetGuardian – Pet Care Monitoring and Support Platform</div>
                        </div>
                    </div>
                    <div class="metrics-row">
                        <div class="metrics-row" style="border:none; padding:0; margin:0; width:100%; display:flex; gap:1.5rem;">
                            <div class="metric-card">
                                <div class="metric-name">Độ tương đồng SW-BTED</div>
                                <div class="metric-value-box">
                                    <span class="metric-value" style="color: var(--accent-red)">0.4264</span>
                                    <span class="metric-status status-crossed">Vượt ngưỡng (0.40)</span>
                                </div>
                            </div>
                            <div class="metric-card">
                                <div class="metric-name">TF-IDF Similarity</div>
                                <div class="metric-value-box">
                                    <span class="metric-value">0.1119</span>
                                </div>
                            </div>
                            <div class="metric-card">
                                <div class="metric-name">SBERT Similarity</div>
                                <div class="metric-value-box">
                                    <span class="metric-value">0.3396</span>
                                </div>
                            </div>
                        </div>
                    </div>
                    <div class="explanation-box">
                        <h4>🔍 Phân tích khoa học</h4>
                        <div class="explanation-text">
                            <strong>Trùng khớp cấu trúc Timeline:</strong> Do lỗi phân loại ngẫu nhiên của bộ sinh dữ liệu (cả hai đề tài đều đạt điểm 0 cho tất cả category và fallback chọn Healthcare), chúng được xếp chung nhóm để làm cặp Type B. Thực tế đây là hai đề tài SE chuẩn hóa có cấu trúc phân chia 6 Task Packages trong Timeline hoàn toàn giống nhau, tạo ra điểm tương đồng Timeline cực cao (<strong>0.5953</strong>), kéo tổng điểm SW-BTED vượt ngưỡng lọc (đạt 0.4264).
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- TAB B1 TF-IDF -->
        <div class="report-section" id="tab-b1-tfidf">
            <!-- Case 1 -->
            <div class="case-card">
                <div class="case-header">
                    <div class="case-title-area">
                        <div class="case-code">Cặp #93</div>
                        <span class="case-type-badge">Type B (Same Domain)</span>
                    </div>
                    <span class="case-error-tag tag-fp">False Positive (Báo động giả)</span>
                </div>
                <div class="case-body">
                    <div class="docs-grid">
                        <div class="doc-info-box">
                            <div class="doc-label">📄 Tài liệu A (Healthcare)</div>
                            <div class="doc-title">SU26SE048: On-demand Home Care Service Application</div>
                        </div>
                        <div class="doc-info-box">
                            <div class="doc-label">📄 Tài liệu B (Healthcare)</div>
                            <div class="doc-title">SU26SE087: PetGuardian – Pet Care Monitoring and Support Platform</div>
                        </div>
                    </div>
                    <div class="metrics-row">
                        <div class="metrics-row" style="border:none; padding:0; margin:0; width:100%; display:flex; gap:1.5rem;">
                            <div class="metric-card">
                                <div class="metric-name">TF-IDF Similarity</div>
                                <div class="metric-value-box">
                                    <span class="metric-value" style="color: var(--accent-red)">0.2394</span>
                                    <span class="metric-status status-crossed">Vượt ngưỡng (0.24)</span>
                                </div>
                            </div>
                            <div class="metric-card">
                                <div class="metric-name">Độ tương đồng SW-BTED</div>
                                <div class="metric-value-box">
                                    <span class="metric-value">0.4001</span>
                                </div>
                            </div>
                            <div class="metric-card">
                                <div class="metric-name">SBERT Similarity</div>
                                <div class="metric-value-box">
                                    <span class="metric-value">0.5087</span>
                                </div>
                            </div>
                        </div>
                    </div>
                    <div class="explanation-box">
                        <h4>🔍 Phân tích khoa học</h4>
                        <div class="explanation-text">
                            <strong>Trùng lặp từ khóa nghiệp vụ:</strong> Do sử dụng chung một loạt các thuật ngữ nghiệp vụ đặc trưng của mô hình chăm sóc theo yêu cầu (booking, care, provider, notification, user, schedule), vector TF-IDF tích lũy điểm tương đồng cao và vượt nhẹ qua ngưỡng tối ưu trung bình 0.24 của mô hình.
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- TAB B2 SBERT -->
        <div class="report-section" id="tab-b2-sbert">
            <!-- Case 1 -->
            <div class="case-card">
                <div class="case-header">
                    <div class="case-title-area">
                        <div class="case-code">Cặp #106</div>
                        <span class="case-type-badge">Type B (Same Domain)</span>
                    </div>
                    <span class="case-error-tag tag-fp">False Positive (Báo động giả)</span>
                </div>
                <div class="case-body">
                    <div class="docs-grid">
                        <div class="doc-info-box">
                            <div class="doc-label">📄 Tài liệu A (Healthcare)</div>
                            <div class="doc-title">SU26SE169: GreenSlot – Smart Urban Vertical Garden Rental Platform...</div>
                        </div>
                        <div class="doc-info-box">
                            <div class="doc-label">📄 Tài liệu B (Healthcare)</div>
                            <div class="doc-title">SP26SE069: Automatic feeding and livestock management system</div>
                        </div>
                    </div>
                    <div class="metrics-row">
                        <div class="metrics-row" style="border:none; padding:0; margin:0; width:100%; display:flex; gap:1.5rem;">
                            <div class="metric-card">
                                <div class="metric-name">SBERT Similarity</div>
                                <div class="metric-value-box">
                                    <span class="metric-value" style="color: var(--accent-red)">0.5606</span>
                                    <span class="metric-status status-crossed">Vượt ngưỡng (0.59)</span>
                                </div>
                            </div>
                            <div class="metric-card">
                                <div class="metric-name">Độ tương đồng SW-BTED</div>
                                <div class="metric-value-box">
                                    <span class="metric-value">0.1370</span>
                                </div>
                            </div>
                            <div class="metric-card">
                                <div class="metric-name">TF-IDF Similarity</div>
                                <div class="metric-value-box">
                                    <span class="metric-value">0.1199</span>
                                </div>
                            </div>
                        </div>
                    </div>
                    <div class="explanation-box">
                        <h4>🔍 Phân tích khoa học</h4>
                        <div class="explanation-text">
                            <strong>Trùng khớp ngữ nghĩa kiến trúc IoT nông nghiệp:</strong> Bản mô tả hệ thống điều khiển tự động và cảm biến giám sát thông minh của cả hai đề tài (làm vườn đô thị và chăn nuôi tự động) chia sẻ các biểu diễn ngữ nghĩa rất gần nhau trong không gian vector của SBERT. Điều này đánh lừa mô hình ngữ nghĩa thô và đẩy điểm số vượt qua ngưỡng phân loại.
                        </div>
                    </div>
                </div>
            </div>
        </div>

    </div>

    <script>
        function switchTab(tabId, btn) {
            // Remove active class from all buttons
            document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
            // Add active class to clicked button
            btn.classList.add('active');
            
            // Hide all report sections
            document.querySelectorAll('.report-section').forEach(s => s.classList.remove('active'));
            // Show selected section
            document.getElementById('tab-' + tabId).classList.add('active');
        }
    </script>
</body>
</html>
"""

    # Write HTML to Report/error_report.html
    report_dir = "Report"
    os.makedirs(report_dir, exist_ok=True)
    report_path = os.path.join(report_dir, "error_report.html")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"Generated error report HTML at: {os.path.abspath(report_path)}")

    # Write HTML to root error_report.html
    root_path = "error_report.html"
    with open(root_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"Generated error report HTML at: {os.path.abspath(root_path)}")

if __name__ == "__main__":
    generate_report()
