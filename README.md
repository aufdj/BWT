# BWT

Burrows-Wheeler Transform in Rust, based on [Mark Nelson's tutorial](https://marknelson.us/posts/1996/09/01/bwt.html).<br>

Transform:<br>
bwt.exe c input output<br>
Inverse Transform:<br>
bwt.exe d input output<br>

<hr>

# fpaq0f-bwt

bwt.exe only applies the transform to data, it doesn't compress it.<br> 
fpaq0f-bwt.exe combines bwt.exe with an adaptive order-0 arithmetic encoder for compression.<br>

Compress:<br>
fpaq0f-bwt.exe c input output<br>
Decompress:<br>
fpaq0f-bwt.exe d input output<br>

[Benchmarks](https://sheet.zoho.com/sheet/open/1pcxk88776ef2c512445c948bee21dcbbdba5?sheet=Sheet1&range=A1)
