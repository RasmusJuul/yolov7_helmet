import sys
import ssl
import smtplib
import mimetypes
from email.message import EmailMessage

timedate = ''
imagepath = ''

try:
    timedate, imagepath = sys.argv[1], sys.argv[2]
except:
    print('error')
    sys.exit(0)

email_sender = 'decoyimg@gmail.com'
email_password = 'sngqzljrmndknmwp'
email_receiver = 'johsgg@gmail.com'

subject = 'DAILY SECURITY BRIEFING'
body = f'A person without a safety helmet was detected at {timedate}'

em = EmailMessage()
em['From'] = email_sender
em['To'] = email_receiver
em['Subject'] = subject
em.set_content(body)

if len(imagepath) > 0:
    mime_type, _ = mimetypes.guess_type(imagepath)
    mime_type, mime_subtype = mime_type.split('/')

    with open(imagepath, 'rb') as file:
        em.add_attachment(file.read(),
        maintype=mime_type,
        subtype=mime_subtype,
        filename=imagepath)


context = ssl.create_default_context()

with smtplib.SMTP_SSL('smtp.gmail.com', 465, context=context) as smtp:
    smtp.login(email_sender, email_password)
    smtp.sendmail(email_sender, email_receiver, em.as_string())

