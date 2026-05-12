import os
import json
import hashlib
import time
from flask import Flask, request, jsonify, session
from flask_cors import CORS
from openai import OpenAI
from dotenv import load_dotenv
from interview_data import QUESTIONS_DB, DIMENSIONS

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-prod')
CORS(app)  # 允许跨域

# ========== DeepSeek 客户端配置 ==========
# DeepSeek 兼容 OpenAI API 格式
DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY')
DEEPSEEK_BASE_URL = os.getenv('DEEPSEEK_BASE_URL', 'https://api.deepseek.com')

# 检查API密钥是否存在
if not DEEPSEEK_API_KEY:
    print("⚠️ 警告: DEEPSEEK_API_KEY 未设置，请在 .env 文件中配置")
    print("   获取方式: https://platform.deepseek.com/")

# 初始化 DeepSeek 客户端
deepseek_client = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL
)

# 使用 DeepSeek 的模型名称
DEEPSEEK_MODEL = "deepseek-chat"  # DeepSeek-V3 模型，性价比最高
# 备选模型（如果需要更强的推理能力，成本稍高）: "deepseek-reasoner"

# ========== 会话存储 ==========
# 生产环境建议使用 Redis，这里用内存存储演示
sessions = {}
# 设置会话过期时间（秒）
SESSION_EXPIRE_SECONDS = 3600  # 1小时

def clean_expired_sessions():
    """清理过期会话（简单实现，可放在定时任务中）"""
    current_time = time.time()
    expired_keys = [
        sid for sid, sess in sessions.items()
        if current_time - sess.get('created_at', 0) > SESSION_EXPIRE_SECONDS
    ]
    for sid in expired_keys:
        del sessions[sid]

# ========== 核心评估函数 ==========
def get_evaluation(question, answer, position):
    """
    调用 DeepSeek API 评估单次回答
    返回格式化的评估结果
    """

    # 获取岗位信息
    position_info = QUESTIONS_DB.get(position, {})
    criteria = position_info.get("scoring_criteria", {})

    # 构建评分标准描述
    criteria_text = "\n".join([f"- {dim}: {criteria.get(dim, '')}" for dim in DIMENSIONS])

    # 构建系统提示词（DeepSeek 推荐使用 system 角色）
    system_prompt = f"""你是一个资深的校招面试官，正在面试一个{position_info.get('name', position)}岗位的应届生。

【评分标准】
{criteria_text}

【任务】
请从{', '.join(DIMENSIONS)}四个维度对候选人的回答进行评分（每个维度1-10分，整数），并给出：
1. 总体评价（2-3句话，总结回答质量）
2. 每个维度的具体评分和简短理由（1句话）
3. 如果是高分（≥7分），给出肯定；如果是低分（≤4分），给出具体改进建议
4. 一个参考的"满分回答示例"（展示面试官期望听到的内容）

请严格以JSON格式输出，不要输出任何其他内容。格式如下：
{{
    "scores": {{
        "专业能力": 0,
        "逻辑表达": 0,
        "实践经验": 0,
        "岗位匹配度": 0
    }},
    "overall_comment": "",
    "dimension_details": {{
        "专业能力": {{"score": 0, "reason": ""}},
        "逻辑表达": {{"score": 0, "reason": ""}},
        "实践经验": {{"score": 0, "reason": ""}},
        "岗位匹配度": {{"score": 0, "reason": ""}}
    }},
    "suggestions": "",
    "perfect_answer": ""
}}"""

    user_prompt = f"""【面试问题】
{question}

【候选人的回答】
{answer}

请按上述要求输出JSON格式的评估结果。"""

    try:
        response = deepseek_client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,  # 较低温度保证一致性
            response_format={"type": "json_object"}  # DeepSeek 支持 JSON 模式
        )

        result_text = response.choices[0].message.content
        result = json.loads(result_text)

        # 确保所有维度都有值
        for dim in DIMENSIONS:
            if dim not in result.get("scores", {}):
                result["scores"][dim] = 5
            if dim not in result.get("dimension_details", {}):
                result["dimension_details"][dim] = {"score": 5, "reason": "暂未提供详细理由"}

        return result

    except Exception as e:
        print(f"DeepSeek API 调用出错: {e}")
        # 返回兜底结果，不中断用户体验
        return {
            "scores": {dim: 5 for dim in DIMENSIONS},
            "overall_comment": "评估服务暂时不可用，请稍后再试。如果问题持续，请检查API密钥配置。",
            "dimension_details": {dim: {"score": 5, "reason": "服务暂不可用，请重试"} for dim in DIMENSIONS},
            "suggestions": "请确保API密钥有效并检查网络连接。",
            "perfect_answer": "参考面试官视角，建议从专业知识、项目经验、思考逻辑几个方面完善回答。"
        }


# ========== API 路由 ==========
@app.route('/')
def index():
    """首页"""
    return app.send_static_file('index.html')
@app.route('/api/positions', methods=['GET'])
def get_positions():
    """获取支持的岗位列表"""
    positions = [
        {"id": pid, "name": info["name"]}
        for pid, info in QUESTIONS_DB.items()
    ]
    return jsonify({"positions": positions})


@app.route('/api/start', methods=['POST'])
def start_interview():
    """开始新的面试，返回第一题"""
    data = request.json
    position = data.get('position')

    if position not in QUESTIONS_DB:
        return jsonify({"error": "不支持的岗位类型"}), 400

    questions = QUESTIONS_DB[position]["questions"]

    # 生成唯一的会话ID
    session_id = hashlib.md5(f"{position}_{time.time()}_{os.urandom(4).hex()}".encode()).hexdigest()

    # 存储会话
    sessions[session_id] = {
        "position": position,
        "questions": questions,
        "current_index": 0,
        "answers": [],
        "evaluations": [],
        "created_at": time.time()
    }

    return jsonify({
        "session_id": session_id,
        "position": position,
        "position_name": QUESTIONS_DB[position]["name"],
        "question": questions[0],
        "current": 1,
        "total": len(questions)
    })


@app.route('/api/submit', methods=['POST'])
def submit_answer():
    """提交当前问题的答案，返回下一题或结束"""
    data = request.json
    session_id = data.get('session_id')
    answer = data.get('answer', '').strip()

    if session_id not in sessions:
        return jsonify({"error": "会话不存在或已过期，请重新开始"}), 404

    session_data = sessions[session_id]
    current_idx = session_data["current_index"]

    # 防止重复提交（如果已经提交过当前问题）
    if current_idx >= len(session_data["answers"]):
        question = session_data["questions"][current_idx]
        position = session_data["position"]

        # 调用 DeepSeek 评估
        evaluation = get_evaluation(question, answer, position)

        # 存储
        session_data["answers"].append({
            "question": question,
            "answer": answer,
            "evaluation": evaluation
        })
        session_data["evaluations"].append(evaluation)

    session_data["current_index"] += 1

    # 判断是否结束
    is_finished = session_data["current_index"] >= len(session_data["questions"])

    if is_finished:
        # 计算总分和各维度平均分
        total_scores = {dim: 0 for dim in DIMENSIONS}
        for ev in session_data["evaluations"]:
            for dim in DIMENSIONS:
                total_scores[dim] += ev["scores"].get(dim, 0)

        avg_scores = {}
        for dim in DIMENSIONS:
            avg_scores[dim] = round(total_scores[dim] / len(session_data["questions"]), 1)

        # 计算总分（四维度平均）
        overall_score = round(sum(avg_scores.values()) / len(DIMENSIONS), 1)

        # 生成综合建议（基于各维度得分）
        weak_dimensions = [dim for dim in DIMENSIONS if avg_scores.get(dim, 0) < 6]
        if weak_dimensions:
            suggestion = f"你的{'、'.join(weak_dimensions)}维度得分较低，建议针对这些方面进行专项练习。可以重新面试同一岗位，或参考参考答案完善回答思路。"
        else:
            suggestion = "你的各项能力表现均衡，表现不错！建议保持练习，并尝试挑战更多岗位的面试。"

        return jsonify({
            "finished": True,
            "session_id": session_id,
            "summary": {
                "total_questions": len(session_data["questions"]),
                "avg_scores": avg_scores,
                "overall_score": overall_score,
                "suggestion": suggestion,
                "evaluations": session_data["evaluations"]
            }
        })
    else:
        # 返回下一题
        next_question = session_data["questions"][session_data["current_index"]]
        current_evaluation = session_data["evaluations"][-1] if session_data["evaluations"] else None

        return jsonify({
            "finished": False,
            "question": next_question,
            "current": session_data["current_index"] + 1,
            "total": len(session_data["questions"]),
            "current_evaluation": current_evaluation
        })


@app.route('/api/evaluation/<session_id>', methods=['GET'])
def get_evaluation_report(session_id):
    """获取完整的面试评估报告"""
    if session_id not in sessions:
        return jsonify({"error": "会话不存在或已过期"}), 404

    session_data = sessions[session_id]

    return jsonify({
        "session_id": session_id,
        "position": session_data["position"],
        "position_name": QUESTIONS_DB[session_data["position"]]["name"],
        "answers": session_data["answers"],
        "evaluations": session_data["evaluations"]
    })


@app.route('/api/health', methods=['GET'])
def health_check():
    """健康检查接口，用于监控"""
    # 检查 DeepSeek API 是否可用（简单测试）
    api_healthy = False
    try:
        # 发送一个极简请求测试
        response = deepseek_client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=5,
            temperature=0
        )
        api_healthy = True
    except Exception as e:
        print(f"API健康检查失败: {e}")

    return jsonify({
        "status": "ok",
        "api_healthy": api_healthy,
        "active_sessions": len(sessions),
        "model": DEEPSEEK_MODEL
    })


# ========== 启动应用 ==========
if __name__ == '__main__':
    print("=" * 50)
    print("AI模拟面试应用启动中...")
    print(f"DeepSeek API: {DEEPSEEK_BASE_URL}")
    print(f"使用模型: {DEEPSEEK_MODEL}")
    print(f"API Key 已配置: {'是' if DEEPSEEK_API_KEY else '否'}")
    print("=" * 50)
    print("访问地址: http://localhost:5000")
    print("按 Ctrl+C 停止服务")
    print("=" * 50)

    app.run(debug=True, port=5000, host='0.0.0.0')
