from psychopy import visual, core, event, gui
import random
import csv
import os

# 실험 정보 입력
dlg = gui.Dlg(title="Decoy Effect Experiment")
dlg.addField("Participant ID:")
info = dlg.show()
if not info:
    core.quit()
participant_id = info[0]

# 데이터 폴더 생성 (없으면)
save_dir = "/Users/youlimkim/Desktop/YLab/Summerschool Poster/Decoy_data"
os.makedirs(save_dir, exist_ok=True)

# 데이터 저장 준비
filename = os.path.join(save_dir, f"decoy_rating_{participant_id}.csv")
with open(filename, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["participant", "target", "candidate", "set", "rating"])

# 윈도우 생성 (전체 화면으로)
win = visual.Window(fullscr=True, color="white", units='norm')

# stimulus 설정
instruction_text = "How similar are the following words to the target word?"
rating_prompt = (
    "How similar is the word below to the target?\n"
    "Please press a number from 1 (Not similar) to 7 (Very similar) on your keyboard."
)
fixation = visual.TextStim(win, text="+", color="black", height=0.15)

# 실험 자극 세트
stimuli = {
    "lacrosse": [
        ("set1", ["hockey", "basketball", "golf"]),
        ("set2", ["hockey", "gateball", "golf"])
    ],
    "sparrow": [
        ("set1", ["flamingo", "bat", "penguin"]),
        ("set2", ["flamingo", "ostriche", "penguin"])
    ],
    "nurse": [
        ("set1", ["paramedic", "engineer", "doctor"]),
        ("set2", ["paramedic", "vet", "doctor"])
    ],
    "orange": [
        ("set1", ["grapefruit", "mango", "lemon"]),
        ("set2", ["grapefruit", "lime", "lemon"])
    ],
    "car": [
        ("set1", ["bus", "plane", "motorbike"]),
        ("set2", ["bus", "bicycle", "motorbike"])
    ]
}

# 실험 실행
for target, sets in stimuli.items():
    for set_name, candidates in sets:
        shuffled = candidates.copy()
        random.shuffle(shuffled)

        # 화면 1: 소개 화면 (단어 한 번에 크게 띄움)
        instruction = visual.TextStim(win, text=instruction_text, pos=(0, 0.7), color="black")
        set_label = visual.TextStim(win, text=f"{target} - {set_name}", pos=(0, 0.55), color="black", height=0.08)
        target_word = visual.TextStim(win, text=target, pos=(0, 0.3), color="black", height=0.15)
        all_words = visual.TextStim(win, text=" / ".join(candidates), pos=(0, -0.2), color="darkblue", height=0.12)
        continue_text = visual.TextStim(win, text="Press any key to continue.", pos=(0, -0.7), color="gray", height=0.05)

        instruction.draw()
        set_label.draw()
        target_word.draw()
        all_words.draw()
        continue_text.draw()
        win.flip()
        event.waitKeys()

        for cand in shuffled:
            # 평가 화면
            prompt = visual.TextStim(win, text=rating_prompt, pos=(0, 0.7), color="black", wrapWidth=1.6)
            target_word = visual.TextStim(win, text=target, pos=(0, 0.4), color="black", height=0.15)
            candidate_word = visual.TextStim(win, text=cand, pos=(0, -0.1), color="black", height=0.12)
            prompt.draw()
            target_word.draw()
            candidate_word.draw()
            win.flip()

            keys = event.waitKeys(keyList=[str(n) for n in range(1, 8)])
            rating = int(keys[0])

            with open(filename, mode="a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow([participant_id, target, cand, set_name, rating])

            # fixation
            fixation.draw()
            win.flip()
            core.wait(1.0)

# 종료 메시지
end_text = visual.TextStim(win, text="Thank you!", color="black")
end_text.draw()
win.flip()
core.wait(3.0)
win.close()
core.quit()

#실행 코드(왜인지 play로는 실행이 안됨): python "/Users/youlimkim/Desktop/YLab/Summerschool Poster/decoy_rating_task_K.py"
# 30 trials 