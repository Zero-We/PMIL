import torch.nn as nn
import torch.nn.functional as F
import torch
from model import resnet34


class IClassifier(nn.Module):
    def __init__(self, mil_model, feature_size=512, out_class=2):
        super(IClassifier, self).__init__()
        temp = resnet34(True)
        temp.fc = nn.Linear(temp.fc.in_features, 2)
        ch = torch.load(mil_model, map_location='cpu')
        temp.load_state_dict(ch['state_dict'])
        self.features = nn.Sequential(*list(temp.children())[:-1])
        self.fc = temp.fc

    def forward(self, features):
        x = self.fc(features)
        return features, x


class BClassifier(nn.Module):
    def __init__(self, num_cluster, feature_size=512, input_size=128, output_class=2, dropout=0):
        super(BClassifier, self).__init__()
        self.transformer = nn.Linear(feature_size, input_size)

        self.classifier = nn.Sequential(
            nn.Linear(num_cluster, 128),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(128, output_class)
        )


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, features, probs, prototype):
        test, m_indices = torch.sort(probs, 0,
                                     descending=True)
        test_array = test.detach().cpu().numpy()
        m_features = torch.index_select(features, dim=0,
                                        index=m_indices[:5, 1])

        ## Metric learning
        f = self.transformer(m_features)
        p = self.transformer(prototype)

        ## Euclidean
        similarity = Euclidean_Similarity(f, p)
        cmax, _ = torch.max(similarity, dim=1, keepdim=True)
        similarity = similarity / cmax

        ## SLN Cosine
        # Cosine
        #similarity = torch.mm(f, p.transpose(0,1))
        #similarity = similarity / torch.norm(f, p=2, dim=1, keepdim=True) / torch.norm(p, p=2)

        coding = torch.mean(similarity, dim=0, keepdim=True)
        x = self.classifier(coding)
        return x, coding


class PBMIL(nn.Module):
    def __init__(self, i_classifier, b_classifier):
        super(PBMIL, self).__init__()
        self.i_classifier = i_classifier
        self.b_classifier = b_classifier

    def forward(self, x, prototype):
        features, score = self.i_classifier(x)
        probs = F.softmax(score, dim=1)
        output, codings = self.b_classifier(features, probs, prototype)
        return score, output, codings


def Euclidean_Similarity(tensor_a, tensor_b):
    device = tensor_a.device
    output = torch.zeros(tensor_a.size(0), tensor_b.size(0), device=device)
    for i in range(tensor_a.size(0)):
        output[i, :] = torch.pairwise_distance(tensor_a[i, :], tensor_b)
    return output