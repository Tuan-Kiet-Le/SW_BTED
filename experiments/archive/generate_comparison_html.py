import json
import os

def generate_html():
    full_texts_path = os.path.join("data", "dataset", "full_texts.json")
    if not os.path.exists(full_texts_path):
        print("Error: full_texts.json not found!")
        return
        
    with open(full_texts_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    target_codes = ["SU26SE102", "SP26SE163", "SP26SE045", "SP26SE071", "SU26SE043"]
    
    html_content = """<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Đối Chiếu Văn Bản Gốc & Paraphrase (GPT-4o-mini)</title>
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
            --sidebar-bg: #111827;
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

        /* Sidebar navigation */
        .sidebar {
            width: 320px;
            background-color: var(--sidebar-bg);
            border-right: 1px solid var(--border-color);
            padding: 2rem 1.5rem;
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
            background: linear-gradient(135deg, #60a5fa, #3b82f6);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .topic-btn {
            background: transparent;
            border: 1px solid transparent;
            color: var(--text-secondary);
            padding: 1rem 1.2rem;
            border-radius: 12px;
            text-align: left;
            font-size: 0.95rem;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            display: flex;
            flex-direction: column;
            gap: 0.25rem;
        }

        .topic-btn:hover {
            background: rgba(255, 255, 255, 0.03);
            color: #fff;
            border-color: rgba(255, 255, 255, 0.05);
        }

        .topic-btn.active {
            background: rgba(59, 130, 246, 0.15);
            border-color: rgba(59, 130, 246, 0.4);
            color: #60a5fa;
            font-weight: 600;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
        }

        .topic-btn span {
            font-size: 0.75rem;
            color: rgba(255, 255, 255, 0.4);
            font-family: monospace;
        }

        /* Main Content container */
        .main-content {
            flex-grow: 1;
            padding: 2.5rem;
            overflow-y: auto;
            max-height: 100vh;
        }

        .header {
            margin-bottom: 2rem;
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 1.5rem;
        }

        .header h1 {
            font-size: 2rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
            letter-spacing: -0.75px;
        }

        .explanation-card {
            background: rgba(239, 68, 68, 0.05);
            border: 1px solid rgba(239, 68, 68, 0.2);
            padding: 1.2rem;
            border-radius: 12px;
            margin-bottom: 2rem;
            line-height: 1.6;
        }

        .explanation-card h3 {
            color: #fca5a5;
            font-size: 1.05rem;
            margin-bottom: 0.5rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        /* Side-by-side View Layout */
        .comparison-container {
            display: none;
            flex-direction: column;
            gap: 2.5rem;
        }

        .comparison-container.active {
            display: flex;
        }

        .section-block {
            background: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            overflow: hidden;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
        }

        .section-header {
            background: rgba(255, 255, 255, 0.02);
            padding: 1.2rem 1.8rem;
            border-bottom: 1px solid var(--border-color);
            font-weight: 600;
            font-size: 1.1rem;
            color: #60a5fa;
            letter-spacing: -0.3px;
        }

        .grid-columns {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1px;
            background-color: var(--border-color);
        }

        .column-content {
            background: var(--bg-color);
            padding: 1.8rem;
            line-height: 1.7;
            font-size: 0.95rem;
            white-space: pre-line;
        }

        .column-title {
            font-weight: 600;
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 1rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .title-orig {
            color: var(--accent-blue);
        }

        .title-plag {
            color: var(--accent-green);
        }

        .empty-text {
            color: rgba(255, 255, 255, 0.25);
            font-style: italic;
        }
    </style>
</head>
<body>

    <!-- Sidebar navigation -->
    <div class="sidebar">
        <div class="sidebar-title">Đề Tài Đối Chiếu</div>
"""
    
    # Render sidebar buttons
    for i, code in enumerate(target_codes):
        active_class = "active" if i == 0 else ""
        semester_name = "Spring Semester 2026" if code.startswith("SP") else "Summer Semester 2026"
        html_content += f"""
        <button class="topic-btn {active_class}" onclick="switchTopic('{code}', this)">
            <span>{code}</span>
            {semester_name}
        </button>
"""
        
    html_content += """
    </div>

    <!-- Main Content -->
    <div class="main-content">
        <div class="header">
            <h1>Đối Chiếu Văn Bản Gốc và Paraphrase</h1>
            <p style="color: var(--text-secondary);">Giao diện trực quan hóa sự khác biệt giữa văn bản đề án tốt nghiệp gốc và bản paraphrase sinh bởi GPT-4o-mini.</p>
        </div>

        <div class="explanation-card">
            <h3>⚠️ Về phản ánh "Cặp bị trùng SU26SE045":</h3>
            <p>Trong danh sách <code>pairs.csv</code>, bạn thấy có 2 dòng tương tự là <strong>SP26SE045</strong> và <strong>SU26SE045</strong>. Đây <strong>không phải là lỗi trùng lặp</strong>, mà là hai đề tài hoàn toàn khác nhau thuộc hai học kỳ khác nhau của Đại học FPT:</p>
            <ul style="margin-left: 1.5rem; margin-top: 0.5rem;">
                <li><strong>SP26SE045</strong>: Đề tài mã số 045 thuộc học kỳ <strong>Spring (SP) 2026</strong>.</li>
                <li><strong>SU26SE045</strong>: Đề tài mã số 045 thuộc học kỳ <strong>Summer (SU) 2026</strong>.</li>
            </ul>
            <p style="margin-top: 0.5rem;">Do đó, hệ thống đã paraphrase độc lập cả hai đề tài này để đưa vào tập kiểm thử, đảm bảo tính khách quan.</p>
        </div>
"""
    
    # Render comparison containers
    for idx, code in enumerate(target_codes):
        active_class = "active" if idx == 0 else ""
        orig_data = data.get(code, {})
        plag_data = data.get(f"{code}_plag", {})
        
        html_content += f"""
        <div class="comparison-container {active_class}" id="container-{code}">
        """
        
        # Sections comparison
        sections = ["Context", "Problem", "Solution", "Theory", "Deliverables", "Methodology", "Timeline", "References"]
        for sec in sections:
            txt_orig = orig_data.get(sec, "").strip()
            txt_plag = plag_data.get(sec, "").strip()
            
            # Skip if both are empty
            if not txt_orig and not txt_plag:
                continue
                
            orig_display = txt_orig if txt_orig else '<span class="empty-text">[Không có dữ liệu]</span>'
            plag_display = txt_plag if txt_plag else '<span class="empty-text">[Không có dữ liệu]</span>'
            
            html_content += f"""
            <div class="section-block">
                <div class="section-header">Chương: {sec}</div>
                <div class="grid-columns">
                    <!-- Cột Gốc -->
                    <div class="column-content">
                        <div class="column-title title-orig">📄 VĂN BẢN GỐC ({code})</div>
                        <div>{orig_display}</div>
                    </div>
                    <!-- Cột Paraphrase -->
                    <div class="column-content">
                        <div class="column-title title-plag">✨ BẢN PARAPHRASE ({code}_plag)</div>
                        <div>{plag_display}</div>
                    </div>
                </div>
            </div>
"""
            
        html_content += """
        </div>
        """
        
    html_content += """
    </div>

    <script>
        function switchTopic(code, btn) {
            // Remove active class from all buttons
            document.querySelectorAll('.topic-btn').forEach(b => b.classList.remove('active'));
            // Add active class to clicked button
            btn.classList.add('active');
            
            // Hide all containers
            document.querySelectorAll('.comparison-container').forEach(c => c.classList.remove('active'));
            // Show selected container
            document.getElementById('container-' + code).classList.add('active');
        }
    </script>
</body>
</html>
"""
    
    # Generate TXT Content
    txt_content = "========================================================================\n"
    txt_content += "ĐỐI CHIẾU VĂN BẢN ĐỀ TÀI GỐC & PARAPHRASE (GPT-4o-mini)\n"
    txt_content += "========================================================================\n\n"
    txt_content += "Lưu ý về các đề tài có mã số giống nhau khác tiền tố:\n"
    txt_content += "- SP26SE045: Spring Semester 2026, Topic 045\n"
    txt_content += "- SU26SE045: Summer Semester 2026, Topic 045\n"
    txt_content += "Đây là các đề tài khác nhau từ hai học kỳ khác nhau.\n\n"
    
    sections = ["Context", "Problem", "Solution", "Theory", "Deliverables", "Methodology", "Timeline", "References"]
    for code in target_codes:
        txt_content += f"========================================================================\n"
        txt_content += f"ĐỀ TÀI: {code}\n"
        txt_content += f"========================================================================\n\n"
        
        orig_data = data.get(code, {})
        plag_data = data.get(f"{code}_plag", {})
        
        for sec in sections:
            txt_orig = orig_data.get(sec, "").strip()
            txt_plag = plag_data.get(sec, "").strip()
            
            if not txt_orig and not txt_plag:
                continue
                
            txt_content += f"------------------------------------------------------------------------\n"
            txt_content += f"CHƯƠNG: {sec}\n"
            txt_content += f"------------------------------------------------------------------------\n"
            txt_content += f"[VĂN BẢN GỐC - {code}]:\n"
            txt_content += f"{txt_orig if txt_orig else '[Không có dữ liệu]'}\n\n"
            txt_content += f"[BẢN PARAPHRASE - {code}_plag]:\n"
            txt_content += f"{txt_plag if txt_plag else '[Không có dữ liệu]'}\n\n"
            
        txt_content += "\n\n"

    # Write HTML
    output_html_path = os.path.join("Report", "comparison_view.html")
    os.makedirs(os.path.dirname(output_html_path), exist_ok=True)
    with open(output_html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
        
    root_html_path = "comparison_view.html"
    with open(root_html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
        
    # Write TXT
    output_txt_path = os.path.join("Report", "comparison_view.txt")
    with open(output_txt_path, "w", encoding="utf-8") as f:
        f.write(txt_content)
        
    root_txt_path = "comparison_view.txt"
    with open(root_txt_path, "w", encoding="utf-8") as f:
        f.write(txt_content)
        
    print(f"Generated comparison HTML at: {os.path.abspath(output_html_path)}")
    print(f"Generated comparison HTML at: {os.path.abspath(root_html_path)}")
    print(f"Generated comparison TXT at: {os.path.abspath(output_txt_path)}")
    print(f"Generated comparison TXT at: {os.path.abspath(root_txt_path)}")

if __name__ == "__main__":
    generate_html()

