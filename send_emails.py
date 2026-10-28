#!/usr/bin/env python3
"""
send_emails.py
A versatile Python script to send emails with attachments to multiple recipients.

Usage:
  1. Edit the CONFIG section below OR pass options via CLI flags.
  2. Run:
     python send_emails.py

CLI Usage Example:
  python send_emails.py \
    --sender "you@gmail.com" \
    --recipients-file recipients.txt \
    --subject "Important Update" \
    --body "Please find the attached document." \
    --attachment "/path/to/document.pdf"
"""

import argparse
import os
import smtplib
import ssl
import sys
import time
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import mimetypes

# ==============================================================================
# CONFIGURATION (Edit these directly, or override via CLI flags / Env vars)
# ==============================================================================
SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", 587))  # 587 for STARTTLS, 465 for SSL

# Sender credentials:
# For Gmail: Use an 'App Password' (https://myaccount.google.com/apppasswords), NOT your regular password.
SENDER_EMAIL = os.getenv("SENDER_EMAIL", "your_email@gmail.com")
SENDER_PASSWORD = os.getenv("SENDER_PASSWORD", "your_app_password_here")

# Default email details (if not provided via CLI):
DEFAULT_SUBJECT = "Your Subject Here"
DEFAULT_BODY = """Hello,

Please find the attached file as requested.

Best regards,
Sender
"""

# Default recipients list (used if no CLI argument is given)
DEFAULT_RECIPIENTS = [
    # "recipient1@example.com",
    # "recipient2@example.com",
]

# Path to the file you want to attach (optional default)
DEFAULT_ATTACHMENT_PATH = ""  # e.g., "document.pdf"

# Delay between sending emails in seconds (helps avoid spam triggers/rate limits)
DELAY_BETWEEN_EMAILS_SEC = 1.0


def create_email_message(sender: str, recipient: str, subject: str, body: str, attachment_path: str = None) -> MIMEMultipart:
    """Constructs the MIME email message with optional attachment."""
    msg = MIMEMultipart()
    msg["From"] = sender
    msg["To"] = recipient
    msg["Subject"] = subject

    # Attach plain text body
    msg.attach(MIMEText(body, "plain", "utf-8"))

    # Attach file if provided
    if attachment_path:
        if not os.path.isfile(attachment_path):
            raise FileNotFoundError(f"Attachment file not found: {attachment_path}")

        filename = os.path.basename(attachment_path)
        content_type, encoding = mimetypes.guess_type(attachment_path)
        if content_type is None or encoding is not None:
            content_type = "application/octet-stream"
        main_type, sub_type = content_type.split("/", 1)

        with open(attachment_path, "rb") as f:
            part = MIMEBase(main_type, sub_type)
            part.set_payload(f.read())
            encoders.encode_base64(part)
            part.add_header(
                "Content-Disposition",
                f'attachment; filename="{filename}"'
            )
            msg.attach(part)

    return msg


def load_recipients_from_file(filepath: str) -> list[str]:
    """Reads recipient email addresses line-by-line from a text or CSV file."""
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Recipients file not found: {filepath}")

    recipients = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            cleaned = line.strip().strip(",")
            if cleaned and not cleaned.startswith("#"):
                recipients.append(cleaned)
    return recipients


def send_batch_emails(
    server_host: str,
    server_port: int,
    sender_email: str,
    sender_password: str,
    recipients: list[str],
    subject: str,
    body: str,
    attachment_path: str = None,
    delay_sec: float = 1.0,
):
    """Connects to SMTP server and sends email to each recipient."""
    if not recipients:
        print("[!] Error: No recipients provided. Aborting.")
        return

    print(f"\n[*] Preparing to send emails:")
    print(f"    - Sender: {sender_email}")
    print(f"    - Recipients count: {len(recipients)}")
    print(f"    - Subject: {subject}")
    print(f"    - Attachment: {attachment_path if attachment_path else 'None'}")
    print(f"    - SMTP Server: {server_host}:{server_port}")
    print()

    # Create secure SSL context
    context = ssl.create_default_context()

    try:
        print("[*] Connecting to SMTP server...")
        if server_port == 465:
            server = smtplib.SMTP_SSL(server_host, server_port, context=context)
        else:
            server = smtplib.SMTP(server_host, server_port)
            server.ehlo()
            server.starttls(context=context)
            server.ehlo()

        print("[*] Logging in...")
        server.login(sender_email, sender_password)
        print("[✓] Login successful!\n")

    except Exception as e:
        print(f"[!] Authentication / Connection failed: {e}")
        return

    success_count = 0
    fail_count = 0

    try:
        for idx, recipient in enumerate(recipients, start=1):
            print(f"[{idx}/{len(recipients)}] Sending to: {recipient} ... ", end="", flush=True)
            try:
                msg = create_email_message(
                    sender=sender_email,
                    recipient=recipient,
                    subject=subject,
                    body=body,
                    attachment_path=attachment_path
                )
                server.send_message(msg)
                print("SUCCESS")
                success_count += 1
            except Exception as exc:
                print(f"FAILED ({exc})")
                fail_count += 1

            if idx < len(recipients) and delay_sec > 0:
                time.sleep(delay_sec)

    finally:
        print("\n[*] Closing connection to SMTP server...")
        server.quit()

    print(f"\n[✓] Done! Sent: {success_count}, Failed: {fail_count}")


def parse_args():
    parser = argparse.ArgumentParser(description="Send emails with attachments to multiple recipients.")
    parser.add_argument("--sender", help="Sender email address", default=SENDER_EMAIL)
    parser.add_argument("--password", help="Sender email password / app password", default=SENDER_PASSWORD)
    parser.add_argument("--smtp-server", help="SMTP server host", default=SMTP_SERVER)
    parser.add_argument("--smtp-port", help="SMTP server port", type=int, default=SMTP_PORT)
    parser.add_argument("--subject", help="Email subject", default=DEFAULT_SUBJECT)
    parser.add_argument("--body", help="Email body text", default=DEFAULT_BODY)
    parser.add_argument("--body-file", help="Path to text file containing email body")
    parser.add_argument("--attachment", help="Path to file to attach", default=DEFAULT_ATTACHMENT_PATH)
    parser.add_argument("--recipients", nargs="*", help="List of recipient emails separated by space")
    parser.add_argument("--recipients-file", help="Path to a text/CSV file with one email per line")
    parser.add_argument("--delay", type=float, help="Delay between emails in seconds", default=DELAY_BETWEEN_EMAILS_SEC)
    return parser.parse_args()


def main():
    args = parse_args()

    # Determine recipients
    recipients = []
    if args.recipients:
        recipients.extend(args.recipients)
    elif args.recipients_file:
        recipients = load_recipients_from_file(args.recipients_file)
    else:
        recipients = DEFAULT_RECIPIENTS

    if not recipients:
        print("[!] No recipients specified. Provide --recipients or --recipients-file, or set DEFAULT_RECIPIENTS.")
        sys.exit(1)

    # Determine email body
    body = args.body
    if args.body_file:
        with open(args.body_file, "r", encoding="utf-8") as f:
            body = f.read()

    # Check attachment if provided
    attachment = args.attachment.strip() if args.attachment else None
    if attachment and not os.path.isfile(attachment):
        print(f"[!] Error: Attachment file '{attachment}' not found.")
        sys.exit(1)

    send_batch_emails(
        server_host=args.smtp_server,
        server_port=args.smtp_port,
        sender_email=args.sender,
        sender_password=args.password,
        recipients=recipients,
        subject=args.subject,
        body=body,
        attachment_path=attachment,
        delay_sec=args.delay,
    )


if __name__ == "__main__":
    main()
