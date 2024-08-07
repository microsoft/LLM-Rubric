from tqdm import tqdm
import numpy as np
import torch
from pydantic.dataclasses import dataclass
from pydantic_numpy import np_array_pydantic_annotated_typing


@dataclass(eq=True, frozen=True)
class Hyperparameters:
    input_size: int
    output_size: int
    num_judges: int
    all_data_size: int
    finetune_output: int = -1
    num_answers: int = 4
    batch_size: int = 32
    learning_rate: float = 0.01
    layer1_size: int = 100
    layer2_size: int = 100
    pretraining_epochs: int = 50
    finetuning_epochs: int = 50
    random_seed: int = 43


@dataclass
class Dataset:
    machine_evaluations: np_array_pydantic_annotated_typing(dimensions=2, data_type=np.float32)
    human_judges: np_array_pydantic_annotated_typing(dimensions=2, data_type=np.int64)
    human_judgments: np_array_pydantic_annotated_typing(dimensions=2, data_type=np.int64)

    def __len__(self) -> int:
        return self.human_judgments.shape[0]

    @property
    def X(self):
        return torch.tensor(self.machine_evaluations)

    @property
    def A(self):
        return torch.tensor(self.human_judges)

    @property
    def Y(self):
        return torch.tensor(self.human_judgments)


class PersonalizedCalibrationNetwork(torch.nn.Module):
    def __init__(
        self, 
        hyperparameters: Hyperparameters,
    ):
        super().__init__()
        self.hyperparameters = hyperparameters

        torch.manual_seed(self.hyperparameters.random_seed)
        self.W1 = torch.nn.Parameter(
            torch.randn(
                (
                    self.hyperparameters.input_size + 1,
                    self.hyperparameters.layer1_size,
                )
            )
        )
        self.W1a = torch.nn.Parameter(
            torch.randn(
                (
                    self.hyperparameters.num_judges,
                    self.hyperparameters.input_size + 1,
                    self.hyperparameters.layer1_size,
                )
            )
            
        )
        self.W2 = torch.nn.Parameter(
            torch.randn(
                (
                    self.hyperparameters.layer1_size + 1,
                    self.hyperparameters.layer2_size,
                ),
            )
        )
        self.W2a = torch.nn.Parameter(
            torch.randn(
                (
                    self.hyperparameters.num_judges,
                    self.hyperparameters.layer1_size + 1,
                    self.hyperparameters.layer2_size,
                ),
            )
        )
        self.V = torch.nn.Parameter(
            torch.randn(
                (
                    self.hyperparameters.output_size,
                    self.hyperparameters.layer2_size + 1,
                    self.hyperparameters.num_answers,
                ),
            )
        )
        self.Va = torch.nn.Parameter(
            torch.randn(
                (
                    self.hyperparameters.num_judges,
                    self.hyperparameters.output_size,
                    self.hyperparameters.layer2_size + 1,
                    self.hyperparameters.num_answers,
                ), 
            )
        )


    def forward(self, X_machine_evals, X_human_judges):
        
        batch_size = X_machine_evals.size(0)
        bias_column = torch.ones((batch_size, 1))

        # Batch size x (input dims + 1)
        input_and_bias = torch.cat([X_machine_evals, bias_column], axis=1)
        W1a_lookup = self.W1a[X_human_judges[:,0]]
        z1_pre = torch.einsum("bi,bio->bo", input_and_bias, self.W1[None, :, :] + W1a_lookup)
        # Batch size x layer1 size 
        z1 = torch.sigmoid(z1_pre)

        z1_and_bias = torch.cat([z1, bias_column], axis=1)
        W2a_lookup = self.W2a[X_human_judges[:,0]]
        z2_pre = torch.einsum("bi,bio->bo", z1_and_bias, self.W2[None, :, :] + W2a_lookup)
        # Batch size x layer2 size 
        z2 = torch.sigmoid(z2_pre)

        z2_and_bias = torch.cat([z2, bias_column], axis=1)
        Va_lookup = self.Va[X_human_judges[:,0]]
        # Batch size x num questions x num answers 
        logits = torch.einsum("bi,bqia->bqa", z2_and_bias, self.V[None, :, :, :] + Va_lookup)
        return logits


    def loss(self, X, A, Y, I=None):
        logits = self.forward(X, A)
        # batch x num answers x num questions  
        permuted_logits = logits.permute(0,2,1)

        if I is not None:
            permuted_logits = permuted_logits[:, :, I]
            Y = Y[:, I]

        xent = torch.nn.functional.cross_entropy(permuted_logits, (Y-1), ignore_index=-1, reduction='none').mean()
        # TODO separate preprocessing of Y out of this function.
        # TODO Switch reduction back to mean so that maked values are properly ignored. 
        return xent

    def decode(self, X, A, I=None):
        logits = self.forward(X, A)
        # batch x num answers x num questions  
        permuted_logits = logits.permute(0,2,1)

        if I is not None:
            permuted_logits = permuted_logits[:, :, I]
        probs = torch.nn.functional.softmax(permuted_logits, dim=1)
        values = torch.tensor([1.0, 2.0, 3.0, 4.0])
        # TODO Support different values or scales. 
        pred = torch.einsum("i,biq->bq", values, probs)
        return pred


def pretrain_loop(model, dataset, optimizer):
        model.train()
        np.random.seed(model.hyperparameters.random_seed)
        pt_steps = model.hyperparameters.pretraining_epochs * int(
            model.hyperparameters.all_data_size / model.hyperparameters.batch_size
        )
        for steps in range(1,  pt_steps + 1):
            # Generate random batch index
            batch_index = np.random.choice(len(dataset), size=model.hyperparameters.batch_size)
            X_machine_evals = dataset.machine_evaluations[batch_index]
            X_human_judges = dataset.human_judges[batch_index]
            Y_human_judgments = dataset.human_judgments[batch_index]
            X_machine_evals_th = torch.tensor(X_machine_evals)
            X_human_judges_th = torch.tensor(X_human_judges)
            Y_human_judgments_th = torch.tensor(Y_human_judgments)
            loss = model.loss(X_machine_evals_th, X_human_judges_th, Y_human_judgments_th)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()


def finetune_loop(model, dataset, optimizer):
        model.train()
        np.random.seed(model.hyperparameters.random_seed)
        ft_steps = model.hyperparameters.finetuning_epochs * int(
            model.hyperparameters.all_data_size / model.hyperparameters.batch_size
        )

        for steps in range(1,  ft_steps + 1):
            # Generate random batch index
            batch_index = np.random.choice(len(dataset), size=model.hyperparameters.batch_size)
            X_machine_evals = dataset.machine_evaluations[batch_index]
            X_human_judges = dataset.human_judges[batch_index]
            Y_human_judgments = dataset.human_judgments[batch_index]
            X_machine_evals_th = torch.tensor(X_machine_evals)
            X_human_judges_th = torch.tensor(X_human_judges)
            Y_human_judgments_th = torch.tensor(Y_human_judgments)
            loss = model.loss(
                X_machine_evals_th, X_human_judges_th, Y_human_judgments_th,
                I=[model.hyperparameters.finetune_output],
                # TODO make this hyperparameter a list
            )
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
