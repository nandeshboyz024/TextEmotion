emotion = ['sadness','anger','love','surprise','fear','joy'];
let toggle = 0;
let show = document.getElementById("show");
show.onclick = ()=>{
    let newDiv = document.getElementById('data_cls');
    if(toggle==0){
        show.textContent = "hide Custom Data";
        toggle=1;
        let itemsData = newDiv.getAttribute('data-items');
        newDiv.innerHTML=itemsData;
        total = itemsData.length;
        datas=[];
        i=1;
        while(i<total){
            let s="";
            if(itemsData[i]=='['){
                i++;
                while(itemsData[i]!=']'){
                    s+=(itemsData[i]);
                    i++;
                }
                let c=parseInt(s.slice(-1));
                s=s.slice(1,-4);
                datas.push([s,c]);
            }
            i++;
        }
        let str0 = `<table>\n`;
        let str1 = `<thead>\n<tr>\n<th>Text</th>\n<th>Label</th>\n</tr>\n</thead>\n`;
        str0=str0+str1;
        for(let i=0;i<datas.length;i++){ 
            let str2 = `<tr>\n<td>${datas[i][0]}</td>\n<td>${emotion[datas[i][1]]}</td>\n</tr>\n`;
            str0 = str0 + str2;
        }
        str0 = str0 + `</table>`;
        newDiv.innerHTML=str0;
        newDiv.style.padding="2px";
        newDiv.style.border="2px solid #dddddd";
    }
    else{
        show.textContent = "Show Custom Data";
        toggle=0;
        newDiv.innerHTML="";
        newDiv.style="none";
        newDiv.style="margin-top:15px"
    }
};

let showmy = document.getElementById("showmy");
showmy.onclick = ()=>{
    let myDiv = document.getElementById("my_data");
    myDiv.classList.toggle("hidden");
    if(showmy.textContent=="Hide My Data"){
        showmy.textContent="Show My Data";
    }
    else{
        showmy.textContent="Hide My Data";
    }
}

const canvas = document.getElementById('cv');
let div = document.getElementById('mydiv');
canvas.width = div.offsetWidth;
canvas.height = div.offsetHeight;

let x1 = Math.random()*(canvas.width-50);
let y1 = Math.random()*(canvas.height-50);
let x2 = Math.random()*(canvas.width-50);
let y2 = Math.random()*(canvas.height-50);
let x3 = Math.random()*(canvas.width-50);
let y3 = Math.random()*(canvas.height-50);
let x4 = Math.random()*(canvas.width-50);
let y4 = Math.random()*(canvas.height-50);
let x5 = Math.random()*(canvas.width-50);
let y5 = Math.random()*(canvas.height-50);
let x6 = Math.random()*(canvas.width-50);
let y6 = Math.random()*(canvas.height-50);

function resizeCanvas(){
    canvas.width = div.offsetWidth;
    canvas.height = div.offsetHeight;
}
var ctx = canvas.getContext('2d');

var image1 = new Image();
var image2 = new Image();
var image3 = new Image();
var image4 = new Image();
var image5 = new Image();
var image6 = new Image();

image1.src='../static/media/joy.gif';
image2.src='../static/media/anger.gif';
image3.src='../static/media/fear.gif';
image4.src='../static/media/love.gif';
image5.src='../static/media/sadness.gif';
image6.src='../static/media/surprise.gif';


var speedX1 = 0.5;
var speedY1 =0.25;
var speedX2 = 0.5;
var speedY2 =0.25;
var speedX3 = 0.5;
var speedY3 =0.25;
var speedX4 = -0.5;
var speedY4 =-0.25;
var speedX5 = -0.5;
var speedY5 =-0.25;
var speedX6 = -0.5;
var speedY6 =-0.25;
function draw(){
    ctx.clearRect(0,0,canvas.width,canvas.height);
    ctx.drawImage(image1,x1,y1,50,50);
    ctx.drawImage(image2,x2,y2,50,50);
    ctx.drawImage(image3,x3,y3,50,50);
    ctx.drawImage(image4,x4,y4,50,50);
    ctx.drawImage(image5,x5,y5,50,50);
    ctx.drawImage(image6,x6,y6,50,50);
    x1=(x1+speedX1+canvas.width)%canvas.width;
    y1=(y1+speedY1+canvas.height)%canvas.height;
    x2=(x2+speedX2+canvas.width)%canvas.width;
    y2=(y2+speedY2+canvas.height)%canvas.height;
    x3=(x3+speedX3+canvas.width)%canvas.width;
    y3=(y3+speedY3+canvas.height)%canvas.height;
    x4=(x4+speedX4+canvas.width)%canvas.width;
    y4=(y4+speedY4+canvas.height)%canvas.height;
    x5=(x5+speedX5+canvas.width)%canvas.width;
    y5=(y5+speedY5+canvas.height)%canvas.height;
    x6=(x6+speedX6+canvas.width)%canvas.width;
    y6=(y6+speedY6+canvas.height)%canvas.height;
    requestAnimationFrame(draw);
}


window.addEventListener('resize',function(){
    resizeCanvas();
    draw();
})
image1.onload = function(){
    canvas.width = div.offsetWidth;
    canvas.height = div.offsetHeight;
    draw();
}
image1.onerror = function(){
    console.error('error');
}
image2.onload = function(){
    canvas.width = div.offsetWidth;
    canvas.height = div.offsetHeight;
    draw();
}
image2.onerror = function(){
    console.error('error');
}
image3.onload = function(){
    canvas.width = div.offsetWidth;
    canvas.height = div.offsetHeight;
    draw();
}
image3.onerror = function(){
    console.error('error');
}
image4.onload = function(){
    canvas.width = div.offsetWidth;
    canvas.height = div.offsetHeight;
    draw();
}
image4.onerror = function(){
    console.error('error');
}
image5.onload = function(){
    canvas.width = div.offsetWidth;
    canvas.height = div.offsetHeight;
    draw();
}
image5.onerror = function(){
    console.error('error');
}
image6.onload = function(){
    canvas.width = div.offsetWidth;
    canvas.height = div.offsetHeight;
    draw();
}
image6.onerror = function(){
    console.error('error');
}

//   const dataUrl = canvas.toDataURL();
//   div.style.backgroundImage = `url(${dataUrl})`;