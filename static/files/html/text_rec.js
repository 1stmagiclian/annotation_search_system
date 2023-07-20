// ip="39.107.97.152"
ip="127.0.0.1"
is_local=0
input_url=""
labels=""
function inputChange() {
  var e = document.getElementById("input");
  if (e.files.length === 0) {
    return
  }
  const file = e.files[0]
  this.file=file
  var url=window.URL.createObjectURL(file)
  var text_img = document.getElementById("text_img");
  text_img.src=url

  is_local=0
}
function tableClick(e) {
  var text_img = document.getElementById("text_img");
  text_img.src=e.src
  input_url=text_img.src

  is_local=1
}
function getSelect(){
    var select = document.getElementById("qicai");
    var options = select.options;      
    var index = select.selectedIndex;       
    var selectedText = options[index].text;


  var select1 = document.getElementById("pre_wenyang");
  var options1 = select1.options;
  var index1 = select1.selectedIndex;
  var selectedText1 = options1[index1].text;
    
    var select2 = document.getElementById("wenyang");
    var options2 = select2.options;      
    var index2 = select2.selectedIndex;       
    var selectedText2 = options2[index2].text;
    
    
    var select3 = document.getElementById("zhidi");
    var options3 = select3.options;      
    var index3 = select3.selectedIndex;       
    var selectedText3 = options3[index3].text;
		getImages(1,selectedText,selectedText1,selectedText2,selectedText3)
	
}
function changePage(pageButton) {

  // var pageText = document.getElementById("pageText");
  localStorage.setItem('page',pageButton.innerHTML);
  var select = document.getElementById("qicai");
  var options = select.options;      
  var index = select.selectedIndex;       
  var selectedText = options[index].text;
  
  
  var select2 = document.getElementById("wenyang");
  var options2 = select2.options;      
  var index2 = select2.selectedIndex;       
  var selectedText2 = options2[index2].text;
    
    
  var select3 = document.getElementById("zhidi");
  var options3 = select3.options;      
  var index3 = select3.selectedIndex;       
  var selectedText3 = options3[index3].text;
  getImages(pageButton.innerHTML,selectedText,selectedText2,selectedText3)

  for(var i=1;i<=20;i++){
    var button = document.getElementById("button"+i);
    if (parseInt(pageButton.innerHTML)<11) {
      button.innerHTML=i;
    }
    else{
      button.innerHTML=parseInt(pageButton.innerHTML)-11+i
    }
  }
}
function getImages(page,equipment,pre_wenyang, patterns,zhidi){
  let formData=new FormData()
  //器材=最左侧
  // 向formData实例中追加属性 (equipment：equipment patterns：patterns)
  formData.append('start',page);
  formData.append('equipment',equipment);
  formData.append('pre_wenyang',pre_wenyang);
  formData.append('patterns',patterns);
  formData.append('zhidi',zhidi);
  const req=new XMLHttpRequest()
  req.onreadystatechange = function(){
    //之前是4
    if(req.readyState===4){
      resObj=JSON.parse(req.response)
      for(var i=0;i<12;i++){
        var text_img = document.getElementById("text_img"+(i+1));
        var text_label = document.getElementById("text_label"+(i+1));
        if(resObj.url[i]==undefined){
          text_img.style.display="none"
          text_label.style.display="none"
        }else{
          text_img.style.display=""
          text_label.style.display=""
          text_img.src=resObj.url[i]
          if(text_label){
            text_label.innerHTML=resObj.labels[i]
          }
        }
      }
      // labels=resObj.labels
    }
  }
  req.open('post','http://'+ip+':8888/getImages',true)
  req.send(formData)
}
function getLabels(){
  let formData=new FormData()
  const req=new XMLHttpRequest()
  req.onreadystatechange = function(){
    if(req.readyState===4){
      labels=JSON.parse(req.response)
      var qicai_select=document.getElementById('qicai');
      var qicai_num=labels[0].length
      for(var i=0;i<qicai_num;i++){
        qicai_select.options.add(new Option(labels[0][i],"0"));
      }
      var wenyang_select=document.getElementById('wenyang');
      var wenyang_num=labels[1].length
      for(var i=0;i<wenyang_num;i++){
        wenyang_select.options.add(new Option(labels[1][i],"0"));
      }
      var zhidi_select=document.getElementById('zhidi');
      var zhidi_num=labels[2].length
      for(var i=0;i<zhidi_num;i++){
        zhidi_select.options.add(new Option(labels[2][i],"0"));
      }
    }
  }
  req.open('post','http://'+ip+':8888/getLabels',true)
  req.send(formData)
}
function upload(){
  if(this.is_local=="0"){
    var render = new FileReader();
    var text_img = document.getElementById("text_img");
    render.readAsDataURL(this.file);
    render.onload = function () {
      this.img_base64=render.result;
      let formData=new FormData()
      // 向formData实例中追加属性 (参数1：属性名 参数2：属性值)
      formData.append('img_base64',this.img_base64);
      formData.append('is_local',is_local);
      const req=new XMLHttpRequest()
      req.onreadystatechange = function(){
        if(req.readyState===4){
          var result_h2 = document.getElementById("result");
          console.log("识别结果："+req.response)
          result_h2.innerHTML="识别结果："+req.response;
        }
      }
      req.open('post','http://'+ip+':8888/classification',true)
      req.send(formData)
      var result_h2 = document.getElementById("result");
      result_h2.innerHTML="识别结果：识别中"
    };
  }else{
    let formData=new FormData()
    // 向formData实例中追加属性 (参数1：属性名 参数2：属性值)
    input_url_len=this.input_url.length
    formData.append('path',this.input_url.substring(8,input_url_len));
    formData.append('is_local',this.is_local);
    const req=new XMLHttpRequest()
    req.onreadystatechange = function(){
      if(req.readyState===4){
        var result_h2 = document.getElementById("result");
        console.log("识别结果："+req.response)
        result_h2.innerHTML="识别结果："+req.response;
      }
    }
    req.open('post','http://'+ip+':8888/classification',true)
    req.send(formData)
    var result_h2 = document.getElementById("result");
    result_h2.innerHTML="识别结果：识别中"
  }
}
function artificialAnnotation(){
  if(localStorage.getItem("username")==null){
    alert("登录后才能标注")
  }else{
    alert("标注成功")
  }
}
function pre_wenyang_change(pre_wenyang_select) {
  var options3 = pre_wenyang_select.options;      
  var index3 = pre_wenyang_select.selectedIndex;       
  var selectedText3 = options3[index3].text;
  var pre_to_details={'动物纹': ['二龙戏珠纹','丹凤纹', '云蝠纹','虎纹','云龙纹','五蝠纹','凤纹','双鱼纹','夔纹', '夔龙纹','大云龙纹','游龙纹','牛纹','璃螭纹','白龙纹', '蝉纹','蟠虯纹','蟠虺纹', '蟠螭纹', '金龙纹','龙纹'],'植物纹': ['串枝花纹', '勾莲纹', '勾莲花卉纹', '四瓣花纹', '团花纹','宝相花纹','折枝桃纹','折枝花卉纹', '折枝花蝶纹', '折枝莲纹', '朵花纹','树皮纹', '桃竹纹', '梅兰纹', '梅纹', '梅花纹', '水仙花纹', '海棠纹','灵芝纹','牡丹纹','百合花纹','石榴纹', '石榴花纹','石竹花纹','胡桃纹', '芍药纹', '芙蓉纹', '芙蓉花纹', '花卉纹','花果纹', '花纹', '荔枝纹','荷叶纹', '荷莲纹','莲实纹', '莲瓣纹', '莲纹', '莲花纹', '菊瓣纹', '菊纹', '菊花纹','萱草纹', '葡萄纹', '葫芦纹', '西番莲纹'],'几何纹': ['双喜字纹', '回纹',  '如意纹', '字纹','寿字纹','弦纹', '棱纹','环带纹','目纹', '锦纹','鳞纹','龟背纹'],'自然纹': ['云纹','山水纹','流水纹','狩猎纹'],'图腾纹': ['二龙戏珠纹','丹凤纹','五蝠纹','兽面纹','凤纹','夔纹', '夔龙纹','大云龙纹','异兽纹','摩竭纹', '暗八仙纹', '涡纹', '游龙纹','璃螭纹','白龙纹', '蟠螭纹', '金龙纹'],'组合纹': ['云雷纹','云龙杂宝纹','八宝纹','凤衔花纹','勾莲花卉纹','双龙捧寿纹','团花蝴蝶纹','寿字缠枝莲九龙纹','寿字花卉纹', '寿字龙凤纹','杂宝纹', '梅兰纹','海水飞兽纹', '牡丹凤凰纹', '瓜蝶纹','祥云团龙纹','花卉双龙捧寿纹', '花卉杂宝纹','花卉麒麟纹','花蝶纹','花鸟纹','草虫纹', '荷莲缠枝牡丹纹','荷莲鸭纹','莲蝠纹', '菊蝶纹虎皮纹','蟠螭百花纹','蟠螭缠枝花卉纹', '蟠螭缠枝花卉莲瓣纹', '鱼藻纹','鸳鸯鹭莲纹', '鹊梅纹', '黑花虎纹', '龙凤纹','龙穿花纹', '龙马纹']};
  var pre_to_detail=pre_to_details[selectedText3];
  var wenyang_select=document.getElementById('wenyang');
  var length=wenyang_select.length-1;
  for(var i=length;i>=1;i--){
    wenyang_select.removeChild(wenyang_select.options[i]);
    wenyang_select.remove(i);
    wenyang_select.options[i]=null;
  }
  for(var i=0;i<pre_to_detail.length;i++){
    wenyang_select.options.add(new Option(pre_to_detail[i],"0"));
  }
}
getLabels()
getImages(1)