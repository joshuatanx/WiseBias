from BiasAsker import *
from apis import *
from multiprocessing.dummy import Process

asker = BiasAsker("en") # for English chatbots, BiasAsker("ch") for Chinese
# asker.initialize_from_file("./new_data/reduced_groups.csv", "./new_data/reduced_sample_bias.csv")
asker.initialize_from_file("./data/groups_for_auto.csv", "./data/sample_bias_data_for_auto.csv")
asker.get_status()

asker_list = BiasAsker.partition(asker, 4)
for asker_slice in asker_list:
    print(asker_slice.get_status())

def ask(asker_slice, bot, bot_name):
    print("begin asking", asker_slice.get_status())
    asker_slice.asking_pair_questions(bot, f"./save/{bot_name}")
    asker_slice.asking_single_questions(bot, f"./save/{bot_name}")

def eval(asker_slice, bot_name):
    asker_slice.pair_test(f"./save/{bot_name}")
    asker_slice.single_test(f"./save/{bot_name}")

bot_name = "blender"
bot = BlenderBot()
proc_list = []
for asker_slice in asker_list:
    proc = Process(target=ask, args=(asker_slice, bot, bot_name))
    proc.start()
    proc_list.append(proc)

for proc in proc_list:
    proc.join()

for asker_slice in asker_list:
    proc = Process(target=eval, args=(asker_slice, bot_name))
    proc.start()
    proc_list.append(proc)

for proc in proc_list:
    proc.join()

for asker_slice in asker_list:
    print(asker_slice.get_status())

asker = BiasAsker.merge(asker_list)
asker.export()