import smtplib
from email.mime.text import MIMEText

smtp_server = "smtp.gmail.com"
smtp_port = 587
smtp_user = "threadstogether23@gmail.com"
smtp_pass = "ccfsvbbatnxasvgs"  # no spaces

msg = MIMEText("Test email from accident detection app.")
msg["Subject"] = "Test Accident Detection Email"
msg["From"] = smtp_user
msg["To"] = "udithsnair@gmail.com"

with smtplib.SMTP(smtp_server, smtp_port) as server:
    server.starttls()
    server.login(smtp_user, smtp_pass)
    server.sendmail(smtp_user, "udithsnair@gmail.com", msg.as_string())
print("Test email sent.")