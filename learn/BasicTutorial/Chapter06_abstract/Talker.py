from abc import abstractmethod, ABC


class Talker(ABC):
    @abstractmethod
    def talk(self):
        pass

    @abstractmethod  # 定义抽象方法，无需实现功能
    def read(self):
        '子类必须定义读功能'
        pass

    @abstractmethod  # 定义抽象方法，无需实现功能
    def write(self):
        '子类必须定义写功能'
        pass