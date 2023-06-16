from torch import nn


class MLP_5linear(nn.Module):
    def __init__(self, in_features_num, middle_features_num, class_num, dropout_p):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(in_features=in_features_num, out_features=middle_features_num),
            nn.Dropout(p=dropout_p),
            nn.ReLU(),
            nn.Linear(in_features=middle_features_num, out_features=middle_features_num),
            nn.Dropout(p=dropout_p),
            nn.ReLU(),
            nn.Linear(in_features=middle_features_num, out_features=middle_features_num),
            nn.Dropout(p=dropout_p),
            nn.ReLU(),
            nn.Linear(in_features=middle_features_num, out_features=middle_features_num),
            nn.Dropout(p=dropout_p),
            nn.ReLU(),
            nn.Linear(in_features=middle_features_num, out_features=class_num)
        )

    def forward(self, x):
        return self.model(x)
