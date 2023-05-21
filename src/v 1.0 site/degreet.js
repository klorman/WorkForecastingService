const hours = new Date().getHours()
let message

if (hours <12)
    message = 'Доброе утро!'
    else message = 'Добрый день!'
    if (hours >16) message = 'Добрый вечер!'

    messageSpan.innerText = message
    
AOS.init(); //aнимации

var x=document.getElementById('login')              //переход между входом и регистрацией
    var y=document.getElementById('register')
    var z=document.getElementById('btn')

    function register(){
        x.style.left = '-450px'
        y.style.left = '50px'
        z.style.left = '110px'
    }
    function login(){
        x.style.left = '50px'
        y.style.left = '450px'
        z.style.left = '0px'
    }

function showPassword(){
        const btn = document.querySelector('.pas-btn')
        const input = document.querySelector('.pas')
        
        btn.addEventListener('click', () => {
            btn.classList.toggle('active')
            if (input.getAttribute('type') === 'password'){
                    input.setAttribute('type', 'text')
                } else{
                    input.setAttribute('type', 'password')
                }
        })
    }
    showPassword()