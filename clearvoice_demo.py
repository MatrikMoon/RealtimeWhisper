from clearvoice.clearvoice import ClearVoice

# Initialize speech enhancement model
cv_se = ClearVoice(
    task='speech_enhancement',
    model_names=['MossFormer2_SE_48K']
)

# Process single audio file
output_wav = cv_se(
    input_path='gura.wav',
    online_write=False
)

# Save enhanced audio
cv_se.write(output_wav, output_path='gura_out.wav')

# cv_sr = ClearVoice(
#     task='speech_super_resolution',
#     model_names=['MossFormer2_SR_48K']
# )

# output_wav = cv_sr(
#     input_path='voice_isolation_demo_out.wav',
#     online_write=False
# )

# # Save enhanced audio
# cv_sr.write(output_wav, output_path='voice_isolation_demo_out_sr.wav')