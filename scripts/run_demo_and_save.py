"""
Script helper để chạy demo và lưu kết quả với UTF-8 encoding
"""
import subprocess
import sys
from pathlib import Path

def run_demo(scale: str, demo_type: str, output_file: str):
    """
    Chạy demo và lưu kết quả vào file với UTF-8 encoding
    
    Args:
        scale: '1k', '10k', hoặc 'full'
        demo_type: 'part1', 'part2', hoặc 'all'
        output_file: Đường dẫn file output
    """
    print(f"🚀 Đang chạy demo {demo_type} với scale {scale}...")
    print(f"📝 Kết quả sẽ được lưu vào: {output_file}")
    print("=" * 70)
    
    # Chạy script gốc
    cmd = [sys.executable, "gqa_reasoning_engine.py", "--scale", scale, "--demo", demo_type]
    
    try:
        # Chạy và capture output
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding='utf-8',
            check=False
        )
        
        # Gộp stdout và stderr
        full_output = result.stdout
        if result.stderr:
            full_output += "\n\n" + "=" * 70 + "\n"
            full_output += "STDERR:\n"
            full_output += result.stderr
        
        # Lưu vào file với UTF-8 encoding
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(full_output)
        
        # Không in full output ra console, chỉ thông báo
        print("\n" + "=" * 70)
        print(f"✅ Đã lưu kết quả vào: {output_file}")
        print(f"📊 File size: {output_path.stat().st_size:,} bytes")
        print(f"⏱️  Exit code: {result.returncode}")
        
        # In 10 dòng cuối của output để preview
        lines = full_output.split('\n')
        if len(lines) > 10:
            print(f"\n📄 Preview (10 dòng cuối):")
            print('\n'.join(lines[-10:]))
        
        return result.returncode
        
    except Exception as e:
        print(f"❌ Error khi chạy demo: {e}")
        return 1

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Chạy demo và lưu kết quả với UTF-8 encoding")
    parser.add_argument("--scale", choices=['1k', '10k', 'full'], required=True, help="Scale của dataset")
    parser.add_argument("--demo", choices=['part1', 'part2', 'all'], required=True, help="Loại demo")
    parser.add_argument("--output", required=True, help="File output")
    
    args = parser.parse_args()
    
    exit_code = run_demo(args.scale, args.demo, args.output)
    sys.exit(exit_code)
