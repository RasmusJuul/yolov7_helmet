import ssl
import smtplib
import mimetypes
from email.message import EmailMessage
import datetime
import os
import glob

def email_sender():
    email_sender = 'decoyimg@gmail.com'
    email_password = 'sngqzljrmndknmwp'
    email_receiver = 'r.juul97@gmail.com'

    subject = 'DAILY SECURITY BRIEFING'
    
    
    date = datetime.datetime.today().strftime('%Y_%m_%d')
    hourlist = os.listdir(f'runs/{date}')
    body = f'Incidents detected on {date}'
    for hour in hourlist:
        with open(f'runs/{date}/{hour}/timestamps.txt', 'r') as fp:
            lines = fp.readlines()
        for line in lines:
            toi = str(datetime.timedelta(seconds=int(line)))
            toi = hour+toi[1:]
            body += f'\n A person without a safety helmet was detected at {toi}'
        
        
    em = EmailMessage()
    em['From'] = email_sender
    em['To'] = email_receiver
    em['Subject'] = subject
    em.set_content(body)
    
    imagepaths = glob.glob(f'runs/{date}/**/*.jpg')
    for imagepath in imagepaths:
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

if __name__ == '__main__':
    email_sender()