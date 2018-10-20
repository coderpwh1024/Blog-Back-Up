const fs = require("fs");
const path = "./";
// var qiniu = require("C:/Users/Administrator/AppData/Roaming/npm/node_modules/qiniu");
var qiniu = require("C:/Users/Administrator/AppData/Roaming/npm/node_modules/qiniu");
var config = new qiniu.conf.Config();
// 空间对应的机房
config.zone = qiniu.zone.Zone_z0;
//需要填写你的 Access Key 和 Secret Key
qiniu.conf.ACCESS_KEY = 'cDbKuhE-1OgoUcQtQrw8kHNImX8I_l4rklPS--d8';
qiniu.conf.SECRET_KEY = 's-M39WD_n8tEm1uP5T3kFML8f3Ghr9nC3X8bXyYc';
	
var mac = new qiniu.auth.digest.Mac(qiniu.conf.ACCESS_KEY, qiniu.conf.SECRET_KEY);
//要上传的空间
bucket = 'shiyanlouphoto';
var options = {
   scope: bucket
}	
//构建上传策略函数
function uptoken(options) {
  var putPolicy = new qiniu.rs.PutPolicy(options);
  return putPolicy.uploadToken(mac);;
}
//构造上传函数
function uploadFile(uptoken, key, localFile) {
    var formUploader = new qiniu.form_up.FormUploader(config);
    var extra = new qiniu.form_up.PutExtra();
    formUploader.putFile(uptoken, key, localFile, extra, function(err, ret) {
      if(!err) {
        // 上传成功， 处理返回值
        console.log('upload success : ',ret.hash, ret.key);
      } else {
        // 上传失败， 处理返回代码
        console.log(err);
      }
  });
}
/**
 * 读取文件后缀名称，并转化成小写
 * @param file_name
 * @returns
 */
function getFilenameSuffix(file_name) {
  if(file_name=='.DS_Store'){
    return '.DS_Store';
  }
    if (file_name == null || file_name.length == 0)
        return null;
    var result = /\.[^\.]+/.exec(file_name);
    return result == null ? null : (result + "").toLowerCase();
}
fs.readdir(path, function (err, files) {
    if (err) {
        return;
    }
    var arr = [];
    (function iterator(index) {
        if (index == files.length) {
            fs.writeFile("./data.json", JSON.stringify(arr, null, "\t"));
            return;
        }
        fs.stat(path + "/" + files[index], function (err, stats) {
            if (err) {
                return;
            }
            if (stats.isFile()) {
              var suffix = getFilenameSuffix(files[index]);
              if(!(suffix=='.js'|| suffix == '.DS_Store'|| suffix == '.json')){
                //要上传文件的本地路径
                filePath = path+'/'+files[index];
                console.log('抓取到文件: '+files[index]);
                //上传到七牛后保存的文件名
                key = files[index];
                //生成上传 Token
                token = uptoken(options);
                // 异步执行
                uploadFile(token, key, filePath);
                arr.push(files[index]);
            }
                      }
            iterator(index + 1);
        })
    }(0));
});