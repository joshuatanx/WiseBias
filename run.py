from BiasAsker import *
from apis import *
from multiprocessing.dummy import Process

asker = BiasAsker("en") # for English chatbots, BiasAsker("ch") for Chinese
asker.initialize_from_file("./new_data/reduced_groups.csv", "./new_data/reduced_sample_bias.csv")
asker.get_status()

asker_list = BiasAsker.partition(asker, 5)
for asker_slice in asker_list:
    print(asker_slice.get_status())

def ask(asker_slice, bot, bot_name):
    print("begin asking", asker_slice.get_status())
    asker_slice.asking_pair_questions(bot, f"./save/{bot_name}")
    asker_slice.asking_single_questions(bot, f"./save/{bot_name}")

bot = BlenderBot()
proc_list = []
for asker_slice in asker_list:
    proc = Process(target=ask, args=(asker_slice, bot, "blender"))
    proc.start()
    proc_list.append(proc)

for proc in proc_list:
    proc.join()

for asker_slice in asker_list:
    print(asker_slice.get_status())