from trl import DPOTrainer

class StepwiseDPOTrainer(DPOTrainer):
    def get_batch_loss_metrics(self, model, batch, train_eval="train"):
        loss, metrics = super().get_batch_loss_metrics(model, batch, train_eval=train_eval)
        metrics["step_avg_loss"] = loss / batch["prompt_input_ids"].shape[0]
        return loss, metrics