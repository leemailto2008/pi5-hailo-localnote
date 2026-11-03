# -*- coding: utf-8 -*-
# hailo_whisper/engine.py
"""
engine.py: Unified Whisper Inference Engine with Hailo-10H NPU Acceleration
and Zero-Downtime CPU Fallback.
Provides seamless integration with voice_pipeline.py and meeting_minutes.py.
"""

import os
import sys
import time
import logging
import re
import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List

logger = logging.getLogger("hailo_whisper.engine")

# Whisper 解碼參數 (HailoRT >= 5.2.0 的 Speech2Text 支援 repetition_penalty 與自動語言偵測)。
#   PI5_WHISPER_LANGUAGE            預設語言 (ISO-639-1，例如 zh、en)；"auto" = 由模型自動偵測
#   PI5_WHISPER_REPETITION_PENALTY  重複懲罰，1.0 = 不懲罰
# 2026-10-29 raspi543 A/B (Whisper-Small HEF，8 s 切片，1/3/5/10 分鐘測試會議＋中英夾雜／英文短句＋底噪)：
#   - repetition_penalty 1.1：平均 CER 0.401 → 0.363，10 分鐘會議的重複迴圈 (同一片段連續 10 次) 消失；1.05／1.15／1.2 以上都較差。
#   - auto：中文會議與強制 zh 結果相同；英文句不再被翻成中文 (CER 0.54 → 0)；底噪片段的幻覺 8/10 → 1–2/10。
WHISPER_LANGUAGE = os.environ.get("PI5_WHISPER_LANGUAGE", "auto").strip().lower() or "auto"
WHISPER_REPETITION_PENALTY = float(os.environ.get("PI5_WHISPER_REPETITION_PENALTY", "1.1"))
# 近乎靜音 (RMS < -55 dBFS) 不送 NPU：
#   - 合成測試：數位靜音被解碼成「有 有，有」。
#   - 228 真人藍牙錄音 (10-29)：說話段 RMS 約 -36～-34 dBFS；錄音尾端無人說話段 -75～-60 dBFS，
#     auto 會在其中幻覺出「I'm going to get this.」「yes The the the」。-55 dBFS 與說話段仍有約 20 dB 餘裕。
SILENCE_RMS = 10 ** (-55 / 20)
# 自動偵測偶爾把中文誤判成日／韓文；輸出含假名或諺文就改用 zh 重跑一次
_UNEXPECTED_SCRIPT_RE = re.compile(r"[\u3040-\u30ff\u31f0-\u31ff\uac00-\ud7af\u1100-\u11ff]")


def is_near_silent(audio: "np.ndarray") -> bool:
    return audio.size == 0 or float(np.sqrt(np.mean(np.square(audio, dtype=np.float64)))) < SILENCE_RMS


def needs_zh_retry(text: str) -> bool:
    return bool(text) and _UNEXPECTED_SCRIPT_RE.search(text) is not None


# 語音偵測 (VAD)：用 faster-whisper 內建的 Silero VAD (CPU 備援路徑本來就以 vad_filter 使用)，找不到語音的切片不送 NPU。
#   PI5_WHISPER_VAD=0 關閉；faster-whisper 未安裝時只用上面的 RMS 門檻。
# 2026-10-29 228 喇叭→藍牙實錄：PipeWire 1.6.9 後錄音電平約高 10 dB，無人說話時底噪約 -49 dBFS，
#   高於 -55 dBFS 門檻，auto 會幻覺出「I'm going to kill you.」之類英文句子並寫進紀要。
#   Silero (預設 threshold 0.5、最短語音 250 ms)：純底噪 13/13 段擋下；真人錄音只擋結尾無人說話段；
#   測試會議壓到 -50 dBFS 再疊真實底噪 (SNR 約 0 dB) 仍 0 段誤擋；每 8 s 切片約 30 ms CPU (單執行緒)。
WHISPER_VAD = os.environ.get("PI5_WHISPER_VAD", "1").strip() != "0"
_VAD_OPTIONS = None


def speech_seconds(audio: "np.ndarray", sr: int = 16000) -> Optional[float]:
    """Silero VAD 偵測到的語音總秒數；VAD 關閉、未安裝或出錯時回傳 None (呼叫端照常送 NPU)。"""
    global _VAD_OPTIONS
    if not WHISPER_VAD:
        return None
    try:
        from faster_whisper.vad import VadOptions, get_speech_timestamps
    except Exception:
        return None
    try:
        if _VAD_OPTIONS is None:
            _VAD_OPTIONS = VadOptions(min_speech_duration_ms=250)
        spans = get_speech_timestamps(np.asarray(audio, dtype=np.float32), vad_options=_VAD_OPTIONS, sampling_rate=sr)
    except Exception as e:
        logger.warning(f"[HailoWhisper] VAD failed, sending the slice anyway: {e}")
        return None
    return sum(s["end"] - s["start"] for s in spans) / float(sr)

# 會議結構詞的系統性同音錯字 (2026-10-30 統計 228 的 51 份逐字稿，約 25.5 萬字)：
#   「X方面，決議是…」：決意 72、捷一 8、決一 3，正確的「決議」只有 4 次。
#   「待辦事項是…」：待辦事像 14、代辦事像 10、代版事像 7、代辦事向 6…，正確的只有 9 次。
# 長會議紀要以「決議」等關鍵字挑出決議條目，錯字會讓決議段落變成「無」(30 分鐘喇叭自測)。
# 只修正後面接「是」的決議，以及「待辦」系列後接「事X」的待辦事項，不動其他用法。
# 10-30 228 再查 61 份逐字稿：還有「待辦事是」(漏了「項」) 6 次、「戴版是向／待辦是向」4 次、「待辦事想」3 次、
# 「待辦職像／待半時像」各 1 次，一併收進來。「事X」照舊直接改；「是X／職X／時X」和單獨的「事」
# 必須後面接著「是」(「待辦事項是…」的句型) 才改，避免「代辦是向國稅局申請」這類正常用法被改掉。
_DECISION_FIX_RE = re.compile(r"(?:決意|捷一|決一|絕一|覺一)(?=\s*是)")
# 11-02 228 95 份逐字稿：「X方面，__是」的 __ 還有「主一／學藝／學一／扇一／捷意／紙一／冶一／卻一／學意／決義／決逸」各 1–2 次，
# 結尾總結開頭還有「最後總結一下，舊一是…」。這些只在「方面」或「總結一下」後面、接著「是」時才改 (「學藝」本身是詞)。
_DECISION_CTX_FIX_RE = re.compile(r"((?:方面|總結一下|最後總結)\s*[，,、]?\s*)"
                                  r"(?:[決捷學主扇紙冶卻覺絕舊隨只職爭][一意義逸藝議])(?=\s*是)")      # 11-03 夜間 30 分鐘：「訓練方面,隨意是」
# 10-31 夜間 30 分鐘喇叭自測又出現「代半事向」「待幹是像是」；84 份逐字稿裡「待板／代半／待幹／太辦／代伴」各 1–2 次，都不是詞，一併收進來
# (「大半」是正常用語，仍不改)。
_TODO_FIX_RE = re.compile(r"(?:待辦|代辦|代版|戴板|戴版|待半|大辦|帶辦|待板|代半|待幹|太辦|代伴|該辦)\s*"
                          r"(?:事\s*[項像相象向下巷想]|(?:[是職時]\s*[項像相象向下巷想]|事)(?=\s*[,，]?\s*是))"
                          r"|大\s*半\s*事\s*[像相象向](?=\s*是)")       # 「待辦事，是蔡組長」「大半事像是林主任」(11-03 夜間 60 分鐘)


# 會議常用詞「品質」「招募」的同音錯字 (10-31 228 84 份逐字稿)：
#   「品值」371 次，正確的「品質」只有 46 次；「招木／招目／招牧／朝目／朝幕／招幕」共約 240 次，「招募」52 次。
#   這些錯字都不是詞 (jieba 詞典查無)，但要避開跨詞的正常組合：
#   「產品值得」「商品值錢」(前一個詞以「品」結尾、後一個詞以「值」開頭)，「朝目標前進」「這一招目前還有效」「幕後」等。
#   45 份會改到的逐字稿：CER 0.3553 → 0.3489，沒有一份變差；逐字稿含議題關鍵字的比例 0.465 → 0.751。
#   「品之」(作品之一、產品之間) 和「公安」(本身是常用詞) 誤改風險高，不處理。
_QUALITY_FIX_RE = re.compile(r"(?<![產商作食藥精物用樣])品\s*值(?![得班錢勤日夜域])")
# 11-02 再補「招牆」「招床」(91 份逐字稿各 3、5 次；229 的 1 分鐘小測也出現)，並排除「牆上、床邊」這類接在後面的詞。
_RECRUIT_FIX_RE = re.compile(r"(?:招\s*[木目牧幕慕暮牆床]|朝\s*[目幕木])"
                             r"(?![前標的光錄睹材工頭板雕匠質場民區師草人羊後僚府牆布上壁角邊根腳面體單沿鋪位子墊])")


# 「第二個議題是…」：「議題」常被寫成「一提／一題／一體／議提／意提／意題…」(10-31 228 84 份逐字稿，
# 「第N個XX是」句型 2168 次中，正確的「議題」878 次、「一提」610、「一題」340、「一體」96、「議提」67…)。
# 只在「第N個」後面、而且接著「是」時才改；「第一題是」(沒有「個」) 這類正常用法不動。
# 11-03 再補 (95 份裡剩下約 130 處)：「第一個一是」23、「讀題」16、「讀」10、「涼體」9、「計」9、「一集」7、「計題」7、「液體」6、
# 「議」6、「體」…；同樣只在「第N個」與「是」之間、而且只認這些固定寫法 (「第二個問題是」「第一個計畫是」「第三個意見是」不動)。
_TOPIC_FIX_RE = re.compile(r"(第\s*[一二三四五六七八九十\d]{1,3}\s*[個格的隔]\s*)"
                           r"(?:一\s*[提題體集期起群]?|[議意亦液讀計涼役]\s*[提題體亦]?|體)(?=\s*[是市示事])")


# 職稱的同音錯字 (11-02 228 95 份逐字稿，約 32 萬字)：負責人要靠職稱才認得出來 (紀要待辦表的 _OWNER_RE)，
#   「工程師」508 次正確，但另有「公成師」138、「工成師」59、「公城市」39、「公城師」28、「工城市」27… 共約 400 次；
#   「課長」只有 17 次正確，「客長」212、「客掌」104；「組長」155 次，「族長／旬長／祖長」77；
#   「協理」195 次，「協利／協裡／協禮／協李」139；「副理」26 次，「負理／負裡／負禮」66。
#   這些錯字都不是詞，只要避開跨詞的正常組合：「工程時間」「做這個工程時」「顧客長期」「乘客」「家族長輩」「負利率」「正負理論」等。
#   「協力」「精力」「經歷」「科長」「富里」是正常用語，不改。
_ENGINEER_FIX_RE = re.compile(r"(?<![辦化加施手分員完開動竣停復電人])"
                              r"(?:[工公宮]\s*[成城承沉陳]\s*[師市失時詩事司]|[工公宮]\s*程\s*[失詩事]|[公宮]\s*程\s*[師市時]"
                              r"|工\s*程\s*時(?=\s*[負主安調掉]))")
_SECTION_CHIEF_FIX_RE = re.compile(r"(?<![顧乘旅遊訪房租常熟新老過貴政駭說刺俠墨劍商來賓酒散主])"
                                   r"(?:[客克]\s*[長棟漲]|[客克課]\s*掌(?!握|控|管|上|心|中|聲|櫃))(?![期久遠大達度年輩高短老])")
_GROUP_LEAD_FIX_RE = re.compile(r"(?<![家民種部氏貴宗皇王上中下初公])[族旬祖主]\s*[長掌](?![期久遠大達度年輩老者握控得官管])")
_ASSOC_MGR_FIX_RE = re.compile(r"(?<![互])協\s*[利裡禮李](?![益率潤息])")
_DEPUTY_MGR_FIX_RE = re.compile(r"(?<![正勝抱擔])負\s*[理裡禮](?![論由想念性財解])")
# 「副理」其他寫法 (父禮 16、付禮 14、父裡 13、富理 11、富里 36、付裡 10…，約 160 次) 多半是正常字詞的一部分 (富里鄉、付禮金)，
# 只在負責人的句型裡改：前面是「請／交給／由／是／安排／有」加一個姓氏字，後面接「負責／主持／會在／跟／調貨／下個月…」。
_DEPUTY_MGR_CTX_FIX_RE = re.compile(r"((?:交給|請|由|是|安排|有)\s*[，,]?\s*(?![在到去來從往的了是和與及])[\u4e00-\u9fff])\s*"
                                    r"[父付富傅負腹福復副傀傽]\s*[理裡禮李利里力]"
                                    r"(?=\s*(?:負|主|會|在|跟|安排|調|掉|下個|下的|回|週|周|[，,。]|第|$))")
# 「經理」也一樣：精理 10、金李 9、金裡 8、經力 8… (「精力」「經歷」本身是詞)，同樣只在負責人句型裡改。
_MANAGER_CTX_FIX_RE = re.compile(r"((?:交給|請|由|是|安排|有)\s*[，,]?\s*(?![在到去來從往的了是和與及大])[\u4e00-\u9fff])\s*"
                                 r"(?!經理)[精驚京晶金經]\s*[理裡禮力利李歷]"
                                 r"(?=\s*(?:負|主|會|在|跟|安排|調|掉|下個|下的|回|週|周|[，,。]|第|$))")
# 「把名單交給黃專，」「交給吳父，」「交給許經，」「交給菜族，」：職稱只剩第一個字 (95 份約 20 次，11-04 夜間 30 分鐘「交給吾父供應商方面」)
_TRUNCATED_TITLE_FIX_RE = re.compile(r"(交給\s*[\u4e00-\u9fff])\s*([專副父經族課協])"
                                     r"(?=\s*(?:[，,。第]|$|供應|展覽|預算|招募|出差|品質|訂單|庫存|[工公]安))")
_TRUNCATED_TITLE = {"專": "專員", "副": "副理", "父": "副理", "經": "經理", "族": "組長", "課": "課長", "協": "協理"}
# 「劉課長主持面是」：主持後面的「面是／面事／面市…」是面試
_HOST_INTERVIEW_FIX_RE = re.compile(r"(主\s*[持吃])\s*面\s*[是市事室世](?!\s*[前後上下])")


# 韌體／良率／驗證的同音錯字 (11-02 228 95 份逐字稿)：
#   「韌體」只有 26 次正確，「任體」310 次 (後面幾乎都接「更新／當機／方面／修正」)，另有「認體／韁體／韍體／韼體…」約 30 次；
#   「良率」489 次，「量率」152、「糧率」15；「驗證」190 次，「驗譁」30、「驗症」15。
#   「任體」只在後面接韌體相關詞時才改 (避開「擔任體育…」)；「量率」要後面接著數字 (「目前量率是 90%」)，
#   並排除「流量率、產量率、含量率…」；「驗症」排除「檢驗症狀」。
_FIRMWARE_FIX_RE = re.compile(r"(?:任|認|韁|韍|韼|訥|訂|韻|韶|韓|響)\s*體(?=\s*(?:更新|當機|方面|修正|升級|版本))")
_YIELD_FIX_RE = re.compile(r"(?:(?<![流產含能質重容數電用銷音存變增總測])量|糧|涼|粒)\s*[率律]"
                           r"(?=\s*(?:是|為|從|大約|約|只有|達到|已經|目前)?\s*(?:百分之|\d))")
_VERIFY_FIX_RE = re.compile(r"驗\s*(?:譁|症)(?!狀|候)")


# 負責人的姓氏同音錯字 (11-03 228 95 份逐字稿，職稱前面那個字)：
#   「劉」42 次正確，「留」215；「李」68，「理」93；「林」160，「零」83、「領」78、「鄰」17；「許」295，「取」56、「緒」17、「緻」8、「續」5；
#   「蔡」16，「菜」36；「楊」50，「洋」28、「陽」8；「黃」206，「皇」20；「吳」5，「無」19。
#   「請」也常被寫成「情」(「情取經理跟對方確認」「情立協理」「情緒經理」)。
#   這些字都不是常見姓氏；只在「字 + 職稱」後面接負責人句型 (負責／主持／會在／跟／安排／調貨／逗號…) 時改，
#   並排除與前一個字成詞的情況 (保留、處理、帶領、錄取、繼續、蔬菜、海洋、太陽、有無…)。
_OWNER_TITLES = r"(?:協理|經理|副理|課長|組長|工程師|專員|主任)"
# 職稱後面直接接下一個議題 (STT 沒斷句：「交給吳副理供應商方面」) 也算負責人句型
_OWNER_FOLLOW = r"(?=\s*(?:負|主|會|在|跟|安排|調|掉|下個|下的|回|送|週|下週|明天|月底|是|[，,。]|第|$|供應|展覽|預算|招募|出差|品質|訂單|庫存|[工公]安))"
_SURNAME_FIX = {"留": "劉", "理": "李", "立": "李", "零": "林", "領": "林", "鄰": "林", "取": "許", "緒": "許", "緻": "許",
                "續": "許", "菜": "蔡", "洋": "楊", "陽": "楊", "皇": "黃", "無": "吳", "吾": "吳"}
_SURNAME_WORD_PREV = {"留": "保停逗挽預居遺殘收拘滯存截自扣去彌", "理": "處管經協副助代總襄治整清辦修護審料受心物道合說真倫地原推條調學打",
                      "立": "成設建獨確樹訂創起站對自中公私國市孤直", "零": "0123456789一二三四五六七八九十〇零點", "領": "帶率引統本占佔認招申首要將綱",
                      "鄰": "近比四毗睦", "取": "錄爭聽採選領獲吸索提擷竊奪換抽攝榨盜騙攫汲考求讀存", "緒": "頭端思心就", "緻": "精細雅別景標興",
                      "續": "繼持陸連後手延接存永", "菜": "蔬點買做炒青白飯配泡野韓台川生", "洋": "海西東遠重崇出留汪大太南北",
                      "陽": "太夕朝歐艷斜陰向重驕", "皇": "教女天堂秦沙法冠", "無": "並毫有絕全從別虛空尚既若亦都也就",
                      "吾": "支伊"}
_OWNER_PLEASE_FIX_RE = re.compile(r"(?<![事感愛友親熱心交劇溫激表同性病民國詳實真神風豪柔疫行內陳求盛深痴色悲苦鄉])情"
                                  r"(?=\s*(?:[\u4e00-\u9fff]\s*)?" + _OWNER_TITLES + _OWNER_FOLLOW + ")")
# 兩個成詞的例外，95 份逐字稿裡全部都是負責人：「預算表清理協理在6月14日前」(請李協理，17 次)、
# 「安排四場面試，有無副理主持」(有吳副理，20 次)。
_PLEASE_LI_FIX_RE = re.compile(r"清\s*理(?=\s*" + _OWNER_TITLES + _OWNER_FOLLOW + ")")
_HAS_WU_FIX_RE = re.compile(r"有\s*無(?=\s*" + _OWNER_TITLES + r"\s*主)")
_SURNAME_FIX_RE = re.compile(r"(?<=(.))?([" + "".join(_SURNAME_FIX) + r"])(?=\s*" + _OWNER_TITLES + _OWNER_FOLLOW + ")")
# 「課長」只剩「客」：「待辦事項是劉客 送預算表」「名單交給留客，」「請留客跟對方確認」(95 份約 40 次)。
# 「留客」(留住客人) 本身是詞，只在指派句型 (交給／請／由／是／有 + 姓 + 客 + 負責人句型的後文) 裡改；顧客、游客、常客等不是姓氏開頭。
_CHIEF_SURNAMES = "劉留柳王李張陳林黃吳蔡楊許鄭謝郭洪曾邱廖賴周徐蘇葉莊呂江何蕭羅潘簡朱鍾彭詹胡施沈余盧梁趙顏翁魏孫戴范宋鄧杜傅侯曹薛卓阮馬董唐藍蔣紀姚連馮程湯田康姜汪鄒黎涂龔韓袁童陸夏邵錢倪溫譚駱熊秦崔尹孔辛陶段葛孟殷賀賈莫"
_CHIEF_STUB_FIX_RE = re.compile(r"((?:交給|請|由|是|有)\s*)([" + _CHIEF_SURNAMES + r"])\s*客\s*[帳廠棘漲脹仗障-]?(?![長掌戶人服廳家運機房棧串座製套觀氣訴])"
                                + _OWNER_FOLLOW)
# 「再增加二位工程師」的「位」寫成「為／維／圍」(95 份逐字稿約 90 次，正確 87)
_HEADCOUNT_FIX_RE = re.compile(r"([\d一二三四五六七八九十兩幾])\s*[為維圍]\s*(?=工程師|專員|主任|同仁|人員|員工|候選人)")
# 「工安」在 95 份逐字稿裡全部寫成同音的「公安」(553 次)，但「公安」本身是常用詞；使用者 11-02 回覆「維持現狀」，不改。
# 「供應商」的第一個字常被寫錯或漏掉 (「依應商」「侶應商」「儂應商」「是應商」)；「對應／回應／反應／適應」後面接商不改。
_SUPPLIER_FIX_RE = re.compile(r"(?<![供反對回適呼響效感因順相答])(?:[依儂侶衣醫宜疑]\s*)?應\s*商(?![品家業機場標量談務人圈店界])")

# 「週五前回覆客戶」95 份逐字稿只有 7 次寫對，另有約 250 次寫成「回不客戶／回副客／回付客戶／回客，」；
# 「溫控模組的客訴有九件」的「客訴」1 次寫對、149 次寫成「客宿／客速」。只在期限 (…前) 後面、或客訴後接件數時改。
_REPLY_CUSTOMER_FIX_RE = re.compile(r"(前\s*)(?:恢\s*)?回\s*(?:[副不夠復付服福腹]\s*客\s*戶|[副不夠復付服福腹覆]?\s*客\s*戶?(?=\s*[，,。第好]|$))")
_COMPLAINT_FIX_RE = re.compile(r"[客刻克特]\s*[宿速素塑溯斯](?=\s*(?:有|是|共|件|方面|案|數|問題|處理|主要|增加|減少|[\d一二三四五六七八九十兩]+\s*[件起筆個案]))")

# 其他高頻同音錯字 (11-03 對照 95 份逐字稿與講稿逐字比對，次數為 15 分鐘以內的會議)：
#   再→在 147 (「沒有問題在全面更新」「下週會在安排」「等試用結果在談」)、試→是／市 111 (「等是用結果」「裝置上市跑3天」)、
#   由→有 45 (「三場面試有許經理主持」)、務→物 43 (「財物部」)、品→平 34 (「平保部」)、客訴→刻速 21、
#   線→現 16 (「產現」)、程→城 21 (「課城」)、仁→人 19 (「五位同人」)、各→個 18 (「請個部門」)、簽→千 18 (「先千八十臺」)、
#   電→店 (「智慧店表」)。都只在固定的前後文裡改：「正在安排」「會在4月5日前」「上市」「有人主持」「財物」本身不動。
_AGAIN_FIX_RE = re.compile(r"(?:(?<=問題)|(?<=結果)|(?<=會)|(?<=之後)|(?<=然後))\s*在(?=\s*(?:全面|談|安排|確認|討論|追蹤|開會|上線))")
_TRIAL_FIX_RE = re.compile(r"(?<=等)\s*[是市事式](?=\s*用\s*結果)|(?<=[臺台]裝置上)\s*[市是事]\s*(?=跑)|(?<=[臺台]設備上)\s*[市是事]\s*(?=跑)")
_HOSTED_BY_FIX_RE = re.compile(r"[有遊](?=\s*[\u4e00-\u9fff]\s*(?:協理|經理|副理|課長|組長|工程師|專員|主任)\s*主\s*[持吃])")
_THIS_MATTER_FIX_RE = re.compile(r"(?<=這件)\s*是\s*(?=由)")
_DEPT_FIX_RE = re.compile(r"財\s*物\s*部|平\s*[保寶]\s*[部圖布步不固]")
_SITE_FIX_RE = re.compile(r"(?<=[產生])\s*現(?=\s*[已發加，,有的])|(?<=課)\s*[城成](?!本)|(?<=位)\s*同\s*人|(?<=請)\s*個\s*(?=部門)|(?<=智慧)\s*店\s*(?=表)")
_SIGN_FIX_RE = re.compile(r"(?:(?<=先)|(?<=是新))\s*千(?=\s*[\d一二三四五六七八九十兩百]+\s*[臺台])")
# 第二批 (同一份逐字比對的下一段)：職缺→直 20、維持→為持／綴持 21、預計→預寄／預及 13、資安→自安 11、閘道器→炸到期／雜到期 19、
# 外殼→外科／外可 10、負責→負負 10、通知→通之 9、替代料→替代了 9、議題是→議題示 9、調貨→掉貨 8、品質異常→品質一長 14、
# 停車場→停車廠 7、八支→八指 6、正式→正是 (切換) 6、文宣→文生 (請) 5、交期→交其 (延) 5、專員→專源 5、意見→一件 (希望) 14
_MISC2_FIX = [
    (re.compile(r"([\d一二三四五六七八九十兩]\s*個\s*)(?:直|職(?![缺位務員責稱場業]))(?=\s*[，,]?\s*(?:已經|已|目前))"), r"\1職缺"),
    # 11-03 夜間 120 分鐘：「下週會再安排六常面是有理協力主持」→ 六場面試由李協理主持
    (re.compile(r"([\d一二三四五六七八九十兩]\s*)[常廠長]\s*(?=面\s*[是市事室世識試])"), r"\1場"),
    (re.compile(r"協\s*力(?=\s*(?:主持|主吃|負責|安排))"), "協理"),
    (re.compile(r"[為綴]\s*持(?=\s*[新原舊現])"), "維持"),
    (re.compile(r"預\s*[寄及](?=\s*[停展上完出切])"), "預計"),
    (re.compile(r"自\s*安(?=\s*[課訓教])"), "資安"),
    (re.compile(r"(?<=工業)\s*[炸雜閘乍]\s*[到道倒]\s*[期器氣其]"), "閘道器"),
    (re.compile(r"外\s*[科可刻](?=\s*模)"), "外殼"),
    (re.compile(r"負\s*負(?!得)(?=\s*[通主安人責,，])"), "負責"),
    (re.compile(r"負\s*(?:[負責]\s*)*責"), "負責"),          # 「負負責責」「負責責」→「負責」
    (re.compile(r"通\s*之(?=\s*所有)"), "通知"),
    (re.compile(r"替代\s*了(?=\s*[持已在正])"), "替代料"),
    (re.compile(r"(?<=議題)\s*示"), "是"),
    (re.compile(r"掉\s*貨"), "調貨"),
    (re.compile(r"品質\s*[一弣異意]\s*[長常場]"), "品質異常"),
    (re.compile(r"停車\s*廠"), "停車場"),
    (re.compile(r"([\d一二三四五六七八九十兩]\s*)指(?=\s*警示)"), r"\1支"),
    (re.compile(r"正\s*是(?=\s*切換)"), "正式"),
    (re.compile(r"文\s*生(?=\s*請)"), "文宣"),
    (re.compile(r"交\s*其(?=\s*延)"), "交期"),
    (re.compile(r"專\s*[源原](?=\s*[會在負主跟安週，,])"), "專員"),
    (re.compile(r"有\s*一件(?=\s*[，,]?\s*希望)"), "有意見"),
]


def _fix_misc_terms(text: str) -> str:
    for pat, rp in _MISC2_FIX:          # 先做 (「六常面是有理協力主持」要先變成「六場…協理」，後面的規則才認得)
        text = pat.sub(rp, text)
    text = _AGAIN_FIX_RE.sub("再", text)
    text = _TRIAL_FIX_RE.sub("試", text)
    text = _THIS_MATTER_FIX_RE.sub("事", _HOSTED_BY_FIX_RE.sub("由", text))
    text = _DEPT_FIX_RE.sub(lambda m: "財務部" if m.group(0).startswith("財") else "品保部", text)

    def site(m):
        g = m.group(0).strip()
        return {"現": "線", "城": "程", "成": "程", "同人": "同仁", "個": "各", "店": "電"}.get(re.sub(r"\s", "", g), g)
    text = _SITE_FIX_RE.sub(site, text)
    text = _SIGN_FIX_RE.sub("簽", text)
    text = re.sub(r"(?<=是)新(?=簽)", "先", text)          # 「決議是新千五十臺」→「先簽」
    return text


def _fix_owner_names(text: str) -> str:
    text = _TRUNCATED_TITLE_FIX_RE.sub(lambda m: m.group(1) + _TRUNCATED_TITLE[m.group(2)], text)
    text = _OWNER_PLEASE_FIX_RE.sub("請", _HAS_WU_FIX_RE.sub("有吳", _PLEASE_LI_FIX_RE.sub("請李", text)))

    def sub(m):
        prev, ch = m.group(1), m.group(2)
        return ch if prev and prev in _SURNAME_WORD_PREV.get(ch, "") else _SURNAME_FIX[ch]
    text = _CHIEF_STUB_FIX_RE.sub(lambda m: f"{m.group(1)}{'劉' if m.group(2) == '留' else m.group(2)}課長", text)
    text = _SURNAME_FIX_RE.sub(sub, text)
    return _HEADCOUNT_FIX_RE.sub(r"\1位", text)


# 招募／訂單句型的同音錯字 (11-02 228 95 份逐字稿)：
#   「已經面試了 12 位候選人」寫成「面世了／面是了／面市了」約 250 次 (正確 102)，「安排三場面試」寫成「三場面是／面事」；
#   「12月3日前完成報到」寫成「完成報道／報導」278 次 (正確 32)；「還剩 8 個名額」寫成「個名兒／個名，」約 200 次；
#   「先簽 50 臺的試單」寫成「市單／示單／是單／事單」約 80 次；「新報價」寫成「新爆價／新爆假」86 次；
#   「全員複訓」寫成「全原附訊／全元付訊／全原復訊…」。
#   「面世」「面市」「報導」本身是詞，只在上述句型裡改 (後面接「N 位」、前面接「N 場」、前面是日期等)。
_INTERVIEW_FIX_RE = re.compile(r"(已\s*經?\s*)面\s*[世是市事室識]\s*了(?=\s*[\d一二三四五六七八九十兩]+\s*位)"
                               r"|([\d一二三四五六七八九十兩]\s*場\s*)面\s*[是市事室世](?=\s*[,，。有由])")
_ONBOARD_FIX_RE = re.compile(r"(日\s*(?:之?前)?\s*)完成\s*報\s*[道導]")
_QUOTA_FIX_RE = re.compile(r"((?:剩|還有|只有)\s*[\d一二三四五六七八九十兩]+\s*個\s*)名\s*(?:兒|鶴)?(?!額)(?=\s*[，,。.請\s]|$)")
_TRIAL_ORDER_FIX_RE = re.compile(r"([臺台]\s*的\s*)[市示是事式誕誤]\s*單|([臺台]\s*)[市示事式]\s*單")
_NEW_QUOTE_FIX_RE = re.compile(r"新\s*爆\s*[價假架]")
_RETRAIN_FIX_RE = re.compile(r"全\s*[原元員源圓]\s*[附付復複富]\s*[訊訓信迅]")
# 「…前完成全員複訓，負責人是…」後半常聽壞或聽丟 (95 份：全原附 20、全原夥 16、全原， 14、全元， 13、全元附 13、全原訊 10…)；
# 只在「完成」後面、接著逗號／句號／負責人／第N個時補成全員複訓 (「完成全部」「完成全原的」不動)
_RETRAIN_TAIL_FIX_RE = re.compile(r"(?<=完成)\s*全\s*(?:[原元員源圓呿]\s*)?(?:[附付復複富夥夠夕]\s*)?(?:[訊訓信迅]\s*)?"
                                  r"(?=\s*[，,。.]|\s*負|\s*第|\s*待|$)")


def _fix_hiring_and_order_terms(text: str) -> str:
    text = _INTERVIEW_FIX_RE.sub(lambda m: f"{m.group(1)}面試了" if m.group(1) is not None else f"{m.group(2)}面試", text)
    text = _QUOTA_FIX_RE.sub(r"\1名額", _ONBOARD_FIX_RE.sub(r"\1完成報到", text))
    text = _TRIAL_ORDER_FIX_RE.sub(lambda m: f"{m.group(1) if m.group(1) is not None else m.group(2)}試單", text)
    return _RETRAIN_TAIL_FIX_RE.sub("全員複訓", _RETRAIN_FIX_RE.sub("全員複訓", _NEW_QUOTE_FIX_RE.sub("新報價", text)))


# 展覽／測試／產品名的同音錯字 (11-02 95 份逐字稿)：「攤位費」寫成「攀位費／攀胃費／胃胃費…」約 150 次 (正確 44)；
# 「展覽」的第二字常變成罕用字 (覃覚覆覵覾覷…) 約 35 次；「測試」寫成「測式」50 次 (排除「預測式」等)；
# 「三輪回歸測試」寫成「三輪回測／回試／回歷測式」；「電源供應器」寫成「電源供印器／電源印器」約 40 次；
# 「溫控模組」寫成「文控模組」102 次 (比正確的 83 次還多)，只在後面接「模組」時改。
_BOOTH_FEE_FIX_RE = re.compile(r"(?:[攀潘潛潮潤盤攜]\s*[位胃未味]|[胃安]\s*胃)\s*費")
_EXPO_FIX_RE = re.compile(r"展\s*[覃覚覆覵覾覷覉覥婭]")
_TEST_FIX_RE = re.compile(r"(?<![預推觀量檢監偵])測\s*式")
_REGRESSION_FIX_RE = re.compile(r"([\d一二三四五六七八九十兩]\s*輪\s*)回\s*[歸龜規彈歷覆覫,，\-]?\s*(?:測\s*[試式測]*|試)")
_PSU_FIX_RE = re.compile(r"電\s*源\s*[供工公]?\s*[印依]\s*器")
_THERMAL_FIX_RE = re.compile(r"(?:[文渾漁問聞灣減渡癮]\s*[控空]|溫\s*[空通同銅]|文\s*[後寇寶])\s*(?=模\s*組)")
# 「天線模組」寫成「天性模組」(9 次)；「品質異常」的「異常」寫成「弈長／弊長／彌長／弄長／益長…」約 65 次、只剩一個字的「品質一，／品質異，」約 60 次
_ANTENNA_FIX_RE = re.compile(r"天\s*性\s*(?=模\s*組)")
_QUALITY_ISSUE_FIX_RE = re.compile(r"品質\s*(?:[弈弊彌弄益弶弧弣異一]\s*[長常嘗]|[一異弈弣弊](?=\s*[，,]))")


def _fix_product_terms(text: str) -> str:
    text = _EXPO_FIX_RE.sub("展覽", _BOOTH_FEE_FIX_RE.sub("攤位費", text))
    text = _REGRESSION_FIX_RE.sub(r"\1回歸測試", text)
    text = _TEST_FIX_RE.sub("測試", text)
    text = _QUALITY_ISSUE_FIX_RE.sub("品質異常", _ANTENNA_FIX_RE.sub("天線", text))
    return _THERMAL_FIX_RE.sub("溫控", _PSU_FIX_RE.sub("電源供應器", text))


# 夾在中文中間的英文碎片 (11-02 95 份逐字稿約 1000 處)：Whisper 會把中文數字寫成英文 (「three 個職缺」「6 weeks」)，
# 或插入沒有意義的字尾 (「持續's 驗證」「至少要做三輪回 its test」「er」「est」)。
# 只動「前後都是中文或中文標點」的小寫字：數字詞改回中文，下列明顯是雜訊的虛詞與字尾刪掉；
# 其他英文 (ERP、test、bug、pin…) 都保留，硬體會議常講。40 份會改到的逐字稿：CER 0.3643 → 0.3615，沒有一份變差。
# 前後 (略過空白) 都必須是中文字或全形標點，英文會議 (2026-11-03 起紀要跟著逐字稿語言) 的「is,」「two,」不會被動到。
# 一次看 1–3 個連在一起的英文字 (「two of 職」)，逐字處理；全部都要保留時整段原樣不動。
_LATIN_IN_CJK_RE = re.compile(r"(?<=[\u4e00-\u9fff，。、])(\s*)((?:'?[a-z][a-z']{0,7}\s*){1,3}?)(?=[\u4e00-\u9fff，。、])")
# 數字前面的雜訊字尾 (「目前er3個職缺」)；只認多字母的雜訊，「v2」「e3」這類型號不動
_LATIN_BEFORE_DIGIT_RE = re.compile(r"(?<=[\u4e00-\u9fff])\s*(?:er|est|ary|ers|ors|ing|um)\s*(?=\d)")
_EN_NUMBERS = {"one": "一", "two": "二", "three": "三", "four": "四", "five": "五", "six": "六", "seven": "七",
               "eight": "八", "nine": "九", "ten": "十"}
_EN_NOISE = {"er", "est", "ed", "y", "v", "es", "e", "en", "ors", "'s", "r", "um", "ers", "ly", "ing", "ar", "al", "nd",
             "le", "ry", "u", "j", "ce", "st", "ong", "et", "the", "is", "to", "in", "on", "of", "an", "as", "he", "its",
             "that", "with", "them", "will", "our", "and", "ary"}


def _fix_latin_in_cjk(text: str) -> str:
    def sub(m):
        words = m.group(2).split()
        out = [_EN_NUMBERS.get(w, w) for w in words if w not in _EN_NOISE]
        if out == words:
            return m.group(0)
        changed = [w for w in out if w not in _EN_NUMBERS.values()]
        joined = "".join(w if w in _EN_NUMBERS.values() else f" {w} " for w in out).strip()
        return joined if not changed else f" {joined} "
    return _LATIN_BEFORE_DIGIT_RE.sub("", _LATIN_IN_CJK_RE.sub(sub, text))


# 英文會議的逐字稿裡，Whisper 偶爾用全形標點斷句（「Okay，let's start」「safety，last week」，11-03 英文 e2e 13 段有 7 處）。
# 整段以英文為主 (英文字母超過中文字 2 倍) 時，全形逗號、句號、冒號、分號、問號、驚嘆號改成半形加空白。
_FULLWIDTH_PUNCT = {"，": ", ", "。": ". ", "：": ": ", "；": "; ", "？": "? ", "！": "! ", "、": ", "}
_FULLWIDTH_RE = re.compile("[" + "".join(_FULLWIDTH_PUNCT) + "]\\s*")


# 中英夾雜的中文會議 (「這個 release 的 schedule 要 delay，QA 的 regression test 還沒 pass」) 英文字母也可能超過中文字 2 倍，
# 但夾在中文裡的英文幾乎沒有功能詞 (the、is、we、to…)；英文句子裡功能詞約佔四成。與 summary_utils.detect_language 同一規則。
_EN_FUNCTION_WORDS = frozenset("the a an is are was were be been we you they he she it i to of and in on for with that this "
                               "these those will would can could should have has had our your their by at from so but not do "
                               "does did there as or if all its let's lets what which who".split())


def _ascii_punct_if_english(text: str) -> str:
    cjk = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
    latin = sum(1 for c in text if c.isascii() and c.isalpha())
    if latin < 20 or latin <= 2 * cjk:
        return text
    if cjk:
        words = re.findall(r"[A-Za-z]+(?:'[a-z]+)?", text)
        share = min(1.0, sum(1 for w in words if w.lower() in _EN_FUNCTION_WORDS) / (0.35 * max(1, len(words))))
        if latin * share <= 2 * cjk:
            return text
    return re.sub(r" {2,}", " ", _FULLWIDTH_RE.sub(lambda m: _FULLWIDTH_PUNCT[m.group(0)[0]], text)).rstrip()


# 日期寫法統一：Whisper 常把「二月十五日」寫成「2.15日」或「2 月 15 日」(10-31 228 69 份逐字稿：分別 119 與 233 次，
# 正常的「2月15日」2230 次)。統一成「2月15日」：24 份會改到的逐字稿，事實召回 0.692 → 0.704 (12 份變好、0 份變差)，CER 沒有變差。
# 只認 1–12 月、1–31 日，且前面不能接數字或小數點 (避免「13.5日」「1.25.3日」之類)。
# 10-31 再補：點的前後有空白、日前面多一個點、補零 (「6. 04. 日」「1.20. 日」「2. 03 日」)，84 份逐字稿約 40 處；
# ≤15 分鐘的 7 份會改到：事實召回 0.689 → 0.692 (3 份變好、0 份變差)，CER 沒有變差。
_DOT_DATE_RE = re.compile(r"(?<![\d.])(1[0-2]|0?[1-9])\s*\.\s*(3[01]|[12]\d|0?[1-9])\s*\.?\s*日")
_SPACED_DATE_RE = re.compile(r"(?<!\d)(1[0-2]|[1-9])\s*月\s*(3[01]|[12]\d|[1-9])\s*日")


def fix_meeting_terms(text: str) -> str:
    if not text:
        return text or ""
    text = _TODO_FIX_RE.sub("待辦事項", _DECISION_FIX_RE.sub("決議", text))
    text = _DECISION_CTX_FIX_RE.sub(r"\1決議", text)
    text = _RECRUIT_FIX_RE.sub("招募", _QUALITY_FIX_RE.sub("品質", text))
    text = _TOPIC_FIX_RE.sub(r"\1議題", text)
    text = _ENGINEER_FIX_RE.sub("工程師", _SECTION_CHIEF_FIX_RE.sub("課長", text))
    text = _ASSOC_MGR_FIX_RE.sub("協理", _GROUP_LEAD_FIX_RE.sub("組長", text))
    text = _DEPUTY_MGR_CTX_FIX_RE.sub(r"\1副理", _DEPUTY_MGR_FIX_RE.sub("副理", text))
    text = _HOST_INTERVIEW_FIX_RE.sub(r"\1面試", _MANAGER_CTX_FIX_RE.sub(r"\1經理", text))
    text = _fix_owner_names(text)
    text = _FIRMWARE_FIX_RE.sub("韌體", text)
    text = _VERIFY_FIX_RE.sub("驗證", _YIELD_FIX_RE.sub("良率", text))
    text = _SUPPLIER_FIX_RE.sub("供應商", text)
    text = _ascii_punct_if_english(_fix_product_terms(_fix_hiring_and_order_terms(_fix_latin_in_cjk(text))))
    # 要在英文碎片拿掉之後 (「前回 er 客，」「第428個 er 體是」)，否則第二次修才會改到
    text = _COMPLAINT_FIX_RE.sub("客訴", _REPLY_CUSTOMER_FIX_RE.sub(r"\1回覆客戶", text))
    text = _TOPIC_FIX_RE.sub(r"\1議題", text)
    text = _fix_owner_names(_fix_misc_terms(text))   # 也在英文碎片拿掉、英文數字改成中文之後 (「two 個直」「請皇 er 專員」)
    text = _HOSTED_BY_FIX_RE.sub("由", _fix_hiring_and_order_terms(text))   # 「六場面是有李協理主持」：前面的修正做完才認得
    text = _DOT_DATE_RE.sub(lambda m: f"{int(m.group(1))}月{int(m.group(2))}日", text)
    return _SPACED_DATE_RE.sub(lambda m: f"{m.group(1)}月{m.group(2)}日", text)


try:
    import opencc
    _OPENCC = opencc.OpenCC('s2twp.json')
except Exception:
    _OPENCC = None


class WhisperEngineInterface(ABC):
    """Abstract Base Class for all Whisper ASR inference backends."""

    @abstractmethod
    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
        initial_prompt: Optional[str] = None,
        temperature: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Transcribe an audio file and return text with performance metrics.

        Returns:
            Dict containing:
                - "text": str (Transcribed and formatted text)
                - "language": str (Detected or specified language)
                - "inference_time_ms": float (Execution latency in ms)
                - "backend": str ("hailo-10h" or "cpu-faster-whisper")
        """
        pass

    @abstractmethod
    def is_hardware_accelerated(self) -> bool:
        """Return True if running on Hailo-10H NPU hardware."""
        pass


class CpuFallbackWhisperEngine(WhisperEngineInterface):
    """
    CPU-based Faster-Whisper fallback engine.
    Ensures zero downtime if Hailo-10H NPU is detached or initializing.
    """

    def __init__(self, model_size: str = "small", device: str = "cpu", compute_type: str = "int8"):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self._model = None

    def _ensure_model(self):
        if self._model is not None:
            return
        try:
            from faster_whisper import WhisperModel
            logger.info(f"[WhisperEngine] Initializing CPU Fallback WhisperModel ({self.model_size}, {self.compute_type})...")
            self._model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
                cpu_threads=4,
            )
            logger.info("[WhisperEngine] CPU Fallback WhisperModel ready.")
        except ImportError:
            logger.warning("[WhisperEngine] faster_whisper not installed in current environment.")
            self._model = None

    def is_hardware_accelerated(self) -> bool:
        return False

    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
        initial_prompt: Optional[str] = None,
        temperature: float = 0.0,
    ) -> Dict[str, Any]:
        self._ensure_model()
        if self._model is None:
            raise RuntimeError("faster_whisper model could not be loaded in CPU fallback.")
        t0 = time.perf_counter()
        segments, info = self._model.transcribe(
            audio_path,
            beam_size=1,
            language=language,
            initial_prompt=initial_prompt,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=400),
            temperature=temperature,
            condition_on_previous_text=False,
        )

        detected_lang = info.language if info else (language or "unknown")
        raw_text = "".join(s.text for s in segments).strip()

        # Traditional Chinese normalization
        final_text = raw_text
        if _OPENCC and detected_lang == "zh":
            final_text = _OPENCC.convert(raw_text)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "text": final_text,
            "language": detected_lang,
            "inference_time_ms": round(elapsed_ms, 2),
            "backend": "cpu-faster-whisper",
        }


_hailo_inferring: bool = False

def is_hailo_inference_active() -> bool:
    """Return True if Hailo-10H NPU is currently executing Whisper speech-to-text inference."""
    return _hailo_inferring


class HailoWhisperEngine(WhisperEngineInterface):
    """
    Hailo-10H NPU (40 TOPS) Accelerated Whisper Inference Engine.
    Leverages native hailo_platform.genai.Speech2Text and on-board 8GB LPDDR4X
    for sub-300ms chunk processing.
    """

    CANDIDATE_HEF_PATHS = [
        "/home/pi/models/hailo10h/Whisper-Small.hef",
        "/home/pi/models/hailo10h/Whisper-Base.hef",
        "/usr/local/hailo/resources/models/Whisper-Base.hef",
        "/home/pi/models/whisper_small_h10h.hef",
    ]

    def __init__(self, hef_path: Optional[str] = None, group_id: Optional[str] = None):
        self.hef_path = hef_path
        # group_id：與 hailo-ollama (HAILO_OLLAMA_VDEVICE_GROUP_ID) 相同時兩個行程可共用 Hailo-10H；None = 獨占 VDevice
        self.group_id = group_id or None
        self._vdevice = None
        self._s2t = None
        self._initialized = False
        self._probe_and_init()

    def _resolve_hef_path(self) -> Optional[str]:
        if self.hef_path and os.path.exists(self.hef_path):
            return self.hef_path
        for p in self.CANDIDATE_HEF_PATHS:
            if os.path.exists(p):
                return p
        return None

    def _probe_and_init(self):
        """Probe HailoRT runtime, VDevice and Speech2Text engine across candidate HEF models."""
        try:
            from hailo_platform import VDevice
            from hailo_platform.genai import Speech2Text
            
            candidates = [self.hef_path] if self.hef_path else self.CANDIDATE_HEF_PATHS
            existing_candidates = [p for p in candidates if p and os.path.exists(p)]
            if not existing_candidates:
                logger.warning("[HailoWhisper] No compatible Whisper HEF model found. Fallback recommended.")
                self._initialized = False
                return

            if self.group_id:
                params = VDevice.create_params()
                params.group_id = self.group_id
                self._vdevice = VDevice(params)
            else:
                self._vdevice = VDevice()
            for cand_path in existing_candidates:
                try:
                    logger.info(f"[HailoWhisper] Attempting Speech2Text initialization with: {cand_path}...")
                    self._s2t = Speech2Text(self._vdevice, cand_path)
                    self.hef_path = cand_path
                    self._initialized = True
                    logger.info(f"✅ [HailoWhisper] Hailo-10H NPU Speech2Text Engine ready with {os.path.basename(cand_path)}!")
                    return
                except Exception as init_err:
                    logger.warning(f"[HailoWhisper] Candidate {cand_path} failed Speech2Text initialization: {init_err}")
            
            self._initialized = False
            logger.warning("[HailoWhisper] All candidate HEF models failed Speech2Text initialization. Fallback recommended.")
        except Exception as e:
            logger.warning(f"[HailoWhisper] Hailo-10H runtime initialization failed: {e}. Fallback recommended.")
            self._initialized = False

    def is_hardware_accelerated(self) -> bool:
        return self._initialized and self._s2t is not None

    def _load_audio_to_pcm_float(self, audio_source: Any) -> Optional[np.ndarray]:
        """Convert audio file or memory buffer to 16kHz float32 [-1.0, 1.0) mono numpy array."""
        import numpy as np
        if isinstance(audio_source, np.ndarray):
            audio = audio_source.astype(np.float32)
            if audio.ndim > 1:
                audio = audio.flatten()
            return np.clip(audio, -1.0, 0.999999)

        if isinstance(audio_source, bytes):
            audio = np.frombuffer(audio_source, dtype=np.int16).astype(np.float32) / 32768.0
            return np.clip(audio, -1.0, 0.999999)

        if isinstance(audio_source, str) and os.path.exists(audio_source):
            try:
                import wave
                with wave.open(audio_source, "rb") as wf:
                    framerate = wf.getframerate()
                    n_channels = wf.getnchannels()
                    frames = wf.readframes(wf.getnframes())
                    audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
                    if n_channels > 1:
                        audio = audio[::n_channels]
                    return np.clip(audio, -1.0, 0.999999)
            except Exception:
                pass

            # ffmpeg fallback
            try:
                import subprocess
                cmd = [
                    "ffmpeg", "-nostdin", "-threads", "1",
                    "-i", audio_source,
                    "-f", "s16le", "-ac", "1", "-acodec", "pcm_s16le", "-ar", "16000", "-"
                ]
                out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL)
                audio = np.frombuffer(out, dtype=np.int16).astype(np.float32) / 32768.0
                return np.clip(audio, -1.0, 0.999999)
            except Exception as e:
                logger.error(f"[HailoWhisper] Failed to load audio via ffmpeg: {e}")
                return None

    def _trim_leading_silence(self, audio: np.ndarray, sr: int = 16000, thresh: float = 0.005) -> np.ndarray:
        """Trim leading silence/background noise to prevent Whisper acoustic hallucination loops."""
        if audio is None or len(audio) < sr:
            return audio
        # 2026-10-22 e2e 實測：把開頭靜音剪到只剩 200 ms 會讓 Whisper 誤判第一個字 (例：「測試的部分」→「estory的部分」)。
        # 改為只在開頭靜音 >= 1.5 s 時修剪，且保留 500 ms 前導；同一測試集 CER 32.5% → 28.9%。
        frame_len = int(sr * 0.1)  # 100ms frames
        max_scan = min(len(audio) - frame_len, sr * 8)  # scan up to first 8 seconds
        for i in range(0, max_scan, frame_len):
            rms = np.sqrt(np.mean(audio[i : i + frame_len] ** 2))
            if rms > thresh:
                if i < int(sr * 1.5):
                    return audio
                lead_idx = max(0, i - int(sr * 0.5))  # preserve 500ms lead-in
                return audio[lead_idx:]
        return audio

    def transcribe(
        self,
        audio_path: Any,
        language: Optional[str] = None,
        initial_prompt: Optional[str] = None,
        temperature: float = 0.0,
        repetition_penalty: Optional[float] = None,
    ) -> Dict[str, Any]:
        if not self.is_hardware_accelerated():
            raise RuntimeError("Hailo-10H Engine not initialized. Please use CPU fallback.")

        global _hailo_inferring
        _hailo_inferring = True
        t0 = time.perf_counter()
        try:
            audio_data = self._load_audio_to_pcm_float(audio_path)
            if audio_data is not None:
                audio_data = self._trim_leading_silence(audio_data)

            if (audio_data is None or len(audio_data) < 1600 or is_near_silent(audio_data)
                    or speech_seconds(audio_data) == 0.0):
                return {
                    "text": "",
                    "language": language or "en",
                    "inference_time_ms": 0.0,
                    "backend": "hailo-10h",
                }

            from hailo_platform.genai import Speech2TextTask
            # 預設 "zh" (中文多位元組 token 解碼較乾淨)；"auto" → None 交給模型自動偵測語言
            lang = (language or WHISPER_LANGUAGE).lower()
            lang_code = None if lang == "auto" else lang
            rp = WHISPER_REPETITION_PENALTY if repetition_penalty is None else float(repetition_penalty)

            # Calculate dynamic timeout: minimum 30s, or 1.5x audio duration + 10s
            audio_dur_sec = len(audio_data) / 16000.0
            calc_timeout_ms = max(30000, int((audio_dur_sec * 1.5 + 10.0) * 1000))

            gen_kwargs: Dict[str, Any] = {
                "audio_data": audio_data,
                "task": Speech2TextTask.TRANSCRIBE,
                "language": lang_code,
                "timeout_ms": calc_timeout_ms,
            }
            if rp != 1.0:   # HailoRT < 5.2.0 沒有這個參數；1.0 時不傳以維持相容
                gen_kwargs["repetition_penalty"] = rp
            raw_text = self._s2t.generate_all_text(**gen_kwargs)
            if lang_code is None and needs_zh_retry(raw_text or ""):
                logger.info(f"[HailoWhisper] auto language produced kana/hangul, retry with zh: {raw_text!r}")
                gen_kwargs["language"] = lang_code = "zh"
                raw_text = self._s2t.generate_all_text(**gen_kwargs)

            final_text = (raw_text or "").strip()
            # Filter prompt bleeding if present
            for filter_word in ("No Hallucination.", "Verbatim speech transcription", "no hallucination", "繁體中文逐字稿", "無幻覺"):
                if filter_word in final_text:
                    final_text = final_text.replace(filter_word, "").strip()

            # Clean Unicode replacement character (\ufffd) and repeated pause noise artifacts
            final_text = final_text.replace("\ufffd", "").replace("\uFFFD", "")
            final_text = re.sub(r"([,\-\s]{2,})+", "，", final_text)
            final_text = re.sub(r"(，\s*){2,}", "，", final_text)
            final_text = re.sub(r"\b(\w)\1{3,}\b", r"\1", final_text)

            # Clean leading/trailing punctuation artifacts (dashes, commas, dots)
            final_text = re.sub(r"^[\s\-,\.，。]+", "", final_text)
            final_text = re.sub(r"[\s\-]+$", "", final_text)

            # Traditional Chinese conversion (Taiwan standard)
            if _OPENCC and final_text:
                try:
                    final_text = _OPENCC.convert(final_text)
                except Exception as cc_err:
                    logger.warning(f"[HailoWhisper] OpenCC conversion warning: {cc_err}")
            final_text = fix_meeting_terms(final_text)

            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            return {
                "text": final_text,
                "language": lang_code or "auto",
                "inference_time_ms": round(elapsed_ms, 2),
                "backend": "hailo-10h",
            }
        finally:
            _hailo_inferring = False


def get_whisper_engine(preferred_backend: str = "auto", hef_path: Optional[str] = None) -> WhisperEngineInterface:
    """
    Factory function to retrieve the optimal Whisper engine based on hardware availability.

    Args:
        preferred_backend: "auto", "hailo", or "cpu".
        hef_path: Custom path to Hailo-10H Whisper HEF model.

    Returns:
        WhisperEngineInterface instance (HailoWhisperEngine or CpuFallbackWhisperEngine).
    """
    if preferred_backend in ("auto", "hailo"):
        try:
            hailo_engine = HailoWhisperEngine(hef_path=hef_path)
            if hailo_engine.is_hardware_accelerated():
                logger.info("[WhisperFactory] Active Backend: Hailo-10H NPU (40 TOPS)")
                return hailo_engine
        except Exception as e:
            logger.warning(f"[WhisperFactory] Hailo-10H detection failed: {e}")

    logger.info("[WhisperFactory] Active Backend: CPU Fallback (faster-whisper)")
    return CpuFallbackWhisperEngine()
