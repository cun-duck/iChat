def optimize_prompt(prompt: str, instructions: str) -> str:
    """
    Mengoptimalkan prompt dengan menambahkan instruksi di depan prompt.
    
    Args:
        prompt (str): Prompt asli.
        instructions (str): Instruksi tambahan untuk mengoptimasi prompt.
    
    Returns:
        str: Prompt yang telah dioptimasi, dengan instruksi (jika ada) diikuti dua baris baru, lalu prompt.
    """
    # Hapus spasi berlebih di awal dan akhir
    prompt = prompt.strip()
    instructions = instructions.strip()
    
    if instructions:
        return f"{instructions}\n\n{prompt}"
    return prompt
