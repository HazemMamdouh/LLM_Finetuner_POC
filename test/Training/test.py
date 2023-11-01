import unittest
import subprocess

class TestFinetuning(unittest.TestCase):

    def test_finetune_model(self):

        saved_model = "test_model_alpaca"
        train_cmd = ["python", "./src/train_bash.py", "--model_name_or_path", "'openlm-research/open_llama_3b_v2'",
                    "--dataset","alpaca_gpt4_en",
                    "--template","default",
                    "--stage","sft",
                    "--do_train",
                    "--finetuning_type","lora",
                    "--lora_target","q_proj,v_proj",
                    "--output_dir", saved_model,
                    "--overwrite_cache",
                    "--per_device_train_batch_size","4",
                    "--gradient_accumulation_steps","4",
                    "--lr_scheduler_type","cosine",
                    "--logging_steps","10",
                    "--save_steps","1000",
                    "--learning_rate","5e-5",
                    "--num_train_epochs","1.0",
                    "--plot_loss",
                    "--fp16"]    
        cmds = ['export CUDA_VISIBLE_DEVICES='+str(gpu_id),
                      ' '.join(train_cmd)]

        process = subprocess.Popen(";".join(cmds), shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        process.wait()
        
        # Check that the model has been finetuned successfully
        self.assertTrue(os.path.exists(saved_model))


if __name__ == '__main__':
    unittest.main()
