import soundfile as sf
from scipy.signal import resample

def wav_load(wav_path, sampling_rate=16000):
    """
    wav_path 파일을 불러와서 sampling_rate로 리샘플링 해줍니다.
    """
    data, samplerate = sf.read(wav_path)
    
    if samplerate != sampling_rate:
        # resample의 두번째 인자는 출력 샘플 수여야 한다.
        # 현재 data의 길이를 기준으로, 타겟 샘플링 레이트에 맞게 리샘플된 길이 계산
        n_samples = int(len(data) * float(sampling_rate) / samplerate)
        data = resample(data, n_samples)
    
    return data

if __name__ == "__main__":
    wav_path = '/workspace/kru_data/train/00000000.wav'
    data = wav_load(wav_path, sampling_rate=16000)

    # 샘플 확인
    print(f"\n로드된 오디오 데이터:")
    print(f"  Shape: {data.shape}")
    print(f"  Dtype: {data.dtype}")
    print(f"  Min: {data.min():.6f}, Max: {data.max():.6f}")