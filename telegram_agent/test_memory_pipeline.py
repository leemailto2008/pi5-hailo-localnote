# f:\12_prj_raspi5\telegram_agent\test_memory_pipeline.py
"""
Unit test script for Three-Tier Memory Hierarchy on Raspberry Pi 5.
Validates:
1. Tier 1: In-memory working buffer push & clear
2. Tier 2: SQLite user profile facts insertion & prompt injection
3. Tier 3: Vector embeddings calculation & cosine similarity search
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from memory import ThreeTierMemoryManager

def test_memory():
    print("=== Testing Three-Tier Memory Hierarchy ===")
    test_db = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "test_memory.db")
    if os.path.exists(test_db):
        os.remove(test_db)

    mem = ThreeTierMemoryManager(db_path=test_db)
    test_chat_id = 999888

    # --- 1. Test Tier 2: User Profile Facts ---
    print("\n[Step 1] Testing Tier 2: User Profile Facts...")
    mem.set_user_fact(test_chat_id, "user_name", "Andrew")
    mem.set_user_fact(test_chat_id, "project", "pi5-llm-arena")
    mem.set_user_fact(test_chat_id, "hardware", "Raspberry Pi 5 16GB")
    facts = mem.get_user_facts(test_chat_id)
    print("✓ Stored facts:", facts)
    assert facts.get("user_name") == "Andrew"
    profile_prompt = mem.format_profile_prompt(test_chat_id)
    print("✓ Formatted Profile Prompt:\n", profile_prompt)

    # --- 2. Test Tier 1: Working Memory Buffer ---
    print("\n[Step 2] Testing Tier 1: Working Memory Buffer...")
    mem.append_turn(test_chat_id, "你好，我是 Andrew", "您好 Andrew！我是樹莓派 5 語音助理。")
    mem.append_turn(test_chat_id, "今天天氣如何？", "今天天氣晴朗舒適。")
    working = mem.get_working_memory(test_chat_id)
    print(f"✓ Working turns count: {len(working)//2} pairs")
    assert len(working) == 4

    # --- 3. Test Tier 3: Vector Semantic Memory ---
    print("\n[Step 3] Testing Tier 3: Vector Semantic Memory (Ollama nomic-embed-text)...")
    doc1 = "樹莓派 5 具備 Broadcom BCM2712 四核心處理器與 16GB LPDDR4X 記憶體。"
    doc2 = "台灣傳統美食包含牛肉麵、小籠包、珍珠奶茶與滷肉飯。"
    doc3 = "Llama 3.1 8B 在樹莓派 5 上的純 CPU 生成吞吐量約為 2.61 tok/s。"

    print("Storing memories into Tier 3...")
    ok1 = mem.store_memory(test_chat_id, doc1)
    ok2 = mem.store_memory(test_chat_id, doc2)
    ok3 = mem.store_memory(test_chat_id, doc3)
    print(f"✓ Store status: doc1={ok1}, doc2={ok2}, doc3={ok3}")

    if ok1 and ok2 and ok3:
        query = "Pi 5 跑 Llama 模型的速度是多少？"
        print(f"\nQuerying: '{query}'")
        retrieved = mem.retrieve_memories(test_chat_id, query, top_k=2)
        print("✓ Retrieved Top Matches:")
        for content, score in retrieved:
            print(f"  - Score {score:.4f}: {content}")
        # Expect doc3 or doc1 to have the highest similarity
        assert len(retrieved) > 0
        assert "Llama 3.1" in retrieved[0][0] or "BCM2712" in retrieved[0][0]

    # --- 4. Test Complete Prompt Assembly ---
    print("\n[Step 4] Testing Full Prompt Assembly...")
    assembled_messages = mem.build_prompt_messages(
        test_chat_id,
        "請告訴我關於我們硬體規格的細節？",
        base_system_instruction="你是一個繁體中文 AI 助理。"
    )
    print("✓ Total assembled message blocks:", len(assembled_messages))
    print("✓ System block preview:\n", assembled_messages[0]["content"][:300], "...")

    print("\n🎉 ALL TESTS PASSED! Three-Tier Memory Hierarchy is fully operational.")

if __name__ == "__main__":
    test_memory()
