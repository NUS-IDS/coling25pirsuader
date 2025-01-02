To run conversation generation loops, provide directory paths to data (example, <a href="../../data/patient_data">patient_data</a>) and 
the openai details in <b>RunConfig</b>.

The <b>DefaultCG</b> does not use dialog act prediction and relies directly on OpenAI model's inherent
capabilities to generate patient-counselor conversation. Therefore, a dialog act model path is not needed.

For conversation generation using predicted dialog acts, use the code in <b>CGWithPredictedActs</b> and specify the
model path in RunConfig.

To train the initial dialog act model, use code in <a href="../dact">dact</a>.<br>
The initial model can be further fine-tuned using rewards using code in <a href="../RL">RL</a>.<br>
The rewards themselves can be learned using code in <a href="../rewards">rewards</a>.


