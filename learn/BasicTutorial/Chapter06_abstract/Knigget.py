from BasicTutorial.Chapter06_abstract.Talker import Talker


class Knigget(Talker):
    def talk(self):
        print("Ni!")

knigget=Knigget();
print(knigget.talk())
