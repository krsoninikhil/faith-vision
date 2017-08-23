# FaithVision

FaithVision is designed to navigate and guide the visually impaired peoples in day to day operations. This version can identify objects in the live mobile camera feed and speak them using an android app.

This is a 24 hour hack developed as team *Faith3.0* during the HackGSF 2017, a hackathon organised by [GSF India](http://www.gsfaccelerator.com/) in Bengaluru. Team members were - [Akhil Ahuja](), [Ritvik Vipra]() and [myself]().

# Implementation Details

- `IP Webcam` android application is used to capture the live video and serve it on a url.
- `ip_webcam.py` keep hitting that url to get a frame at a moment and upload it to a EC2 instance. `~/.ssh/UbuntuServerPrivateKey.pem` is the key for connecting to that instance.
- `main_script.py` runs on that EC2 instance which process that frame image using Tensorflow's Object Detection APIs and write the results on a file `text.json` (using a file for directly storing makes it unfit for multiple users, but this was quickest way for hackathon).
- `android/FaithVision` (an android app) hits the url serving that json file from the server instance. It then send the parsed text to the `AWS Polly` which returns the audio in `.mp3` and plays it. The app is just the accordingly modified version of the demo applicatioin available on the Polly's site.

# Instructions to run

1. Open `IP Webcam` app and click on start server
2. Run `ip_webcam.py` on a machine (on same network)
3. Run tensorflow script on EC2 instance.
4. Open FaithVision android app for demo.

# Licence

[MIT License](https://nks.mit-license.org/)
