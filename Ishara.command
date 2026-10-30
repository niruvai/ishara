#!/bin/bash
# ডাবল-ক্লিক করলেই ইশারা চালু হবে
cd "$(dirname "$0")"
if [ ! -d venv ]; then
  echo "প্রথমবার সেটআপ হচ্ছে..."
  python3 -m venv venv
  ./venv/bin/pip install -q -r requirements.txt
fi
./venv/bin/python ishara.py "$@"
