import torch
class Config:
    def __init__(self, args) -> None:
        self.__dict__ = args.__dict__
        # self.seed = args.seed
        # self.iterations = args.iterations
        # self.horizon = args.horizon
        # self.arms = args.arms
        # self.points = args.points