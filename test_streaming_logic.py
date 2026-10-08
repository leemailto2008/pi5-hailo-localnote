# f:\12_prj_raspi5\test_streaming_logic.py
import time
import asyncio

async def simulate_streaming(duration_sec=30):
    print(f"=== 開始模擬 {duration_sec} 秒串流狀態機驗證 ===")
    start_time = time.time()
    
    # 30秒錄音，每6秒一切片，預期切出 5 個切片
    total_chunks = 5
    print(f"預期切片數: {total_chunks} 塊")
    
    processed_chunks = set()
    chunk_idx = 0
    
    # 模擬檔案生成時間表
    # 6s: chunk_0, 12s: chunk_1, 18s: chunk_2, 24s: chunk_3, 30s: chunk_4
    file_birth = {0: 1.0, 1: 2.0, 2: 3.0, 3: 4.0, 4: 5.0}
    
    is_recording = True
    ffmpeg_done = False
    
    while True:
        sim_elapsed = time.time() - start_time
        # 模擬 5 秒到達 (縮比代表 30 秒)
        time_exceeded = (sim_elapsed >= 5.0)
        
        if time_exceeded:
            ffmpeg_done = True
            
        if ffmpeg_done and is_recording:
            is_recording = False
            print(f"[{sim_elapsed:.2f}s] 錄音結束！觸發 recording_done 事件")
            
        curr_exists = (chunk_idx in file_birth and sim_elapsed >= file_birth[chunk_idx])
        next_exists = ((chunk_idx + 1) in file_birth and sim_elapsed >= file_birth[chunk_idx + 1])
        
        chunk_ready = False
        if curr_exists and chunk_idx not in processed_chunks:
            if next_exists or ffmpeg_done:
                chunk_ready = True
                
        if chunk_ready:
            processed_chunks.add(chunk_idx)
            print(f"[{sim_elapsed:.2f}s] -> 開始轉錄分塊 {chunk_idx + 1}/{total_chunks}...")
            await asyncio.sleep(0.3)
            print(f"[{time.time() - start_time:.2f}s] -> 分塊 {chunk_idx + 1} 轉錄完成！")
            chunk_idx += 1
            continue
            
        if ffmpeg_done and chunk_idx >= total_chunks:
            print(f"[{sim_elapsed:.2f}s] 全部 {total_chunks} 個分塊轉錄完成！順暢交接 Qwen 英文口譯！")
            break
            
        await asyncio.sleep(0.05)

if __name__ == '__main__':
    asyncio.run(simulate_streaming(30))
