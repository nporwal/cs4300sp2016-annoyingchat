from django.shortcuts import render
from django.shortcuts import render_to_response
from django.http import HttpResponse
from .models import Docs
from django.template import loader
from .form import QueryForm
from project_template import qf
from project_template.twitter import get_tweet
from django.core.paginator import Paginator, EmptyPage, PageNotAnInteger


# Create your views here.
def index(request):
    output_list = ''
    output = ''
    algorithm = ''
    search = ''
    twitter = ''
    if request.GET.get('search'):
        search = request.GET.get('search')
        algorithm = request.GET.get('algorithm')
        if algorithm == "random":
            output_list = qf.find_random()
        elif algorithm == "trivial":
            output_list = qf.find_similar(search)
        elif algorithm == "final":
            output_list = qf.find_final(search)
        paginator = Paginator(output_list, 10)
        page = request.GET.get('page')
        try:
            output = paginator.page(page)
        except PageNotAnInteger:
            output = paginator.page(1)
        except EmptyPage:
            output = paginator.page(paginator.num_pages)
    if request.GET.get('twitter'):
        twitter = request.GET.get('twitter')
        output_list = [get_tweet()]
        paginator = Paginator(output_list, 10)
        page = request.GET.get('page')
        try:
            output = paginator.page(page)
        except PageNotAnInteger:
            output = paginator.page(1)
        except EmptyPage:
            output = paginator.page(paginator.num_pages)
    return render_to_response('project_template/index.html',
                          {'output': output,
                           'algorithm': algorithm,
                           'magic_url': request.get_full_path(),
                           'search' : search
                           })
