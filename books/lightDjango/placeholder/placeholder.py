import sys

### settings
from django.conf import settings

settings.configure(
	DEBUG=True,
	SECRET_KEY='MoryAndAnn',
	ROOT_URLCONF=__name__,
	MIDDLEWARE_CLASSES=(
		'django.middleware.common.CommonMiddleware',
		'django.middleware.csrf.CsrfViewMiddleware',
		'django.middleware.clickjacking.XFrameOptionsMiddleware',
	),
)


### urls and views, forms
from django.conf.urls import url
from django.http import HttpResponse, HttpResponseBadRequest
from django import forms

from io import BytesIO
from PIL import Image

class ImageForm(forms.Form):
	"""Form to validate requested placeholder image"""
	
	height = forms.IntegerField(min_value=1, max_value=2000)
	width = forms.IntegerField(min_value=1, max_value=2000)
	
	def generate(self, image_format='PNG'):
		"""Generate an image of the given type and return as row bytes"""
		height = self.cleaned_data['height']
		width = self.cleaned_data['width']
		image = Image.new('RGB', (width, height), color="#FF0000")
		content = BytesIO()
		image.save(content, image_format)
		content.seek(0)
		return content
	
	
def placeholder(request, width, height):
	form = ImageForm({'height': height, 'width': width})
	if form.is_valid():
		image = form.generate()
		return HttpResponse(image, content_type='image/png')
	else:
		return HttpResponseBadRequest('Invalid image Request')


def index(request):
	return HttpResponse('Hello World')
	
urlpatterns = (
	url(r'^image/(?P<width>[0-9]+)x(?P<height>[0-9]+)/$', placeholder, name="placeholder"),
	url(r'^$', index),
)


if __name__ == '__main__':
	from django.core.management import execute_from_command_line
	
	execute_from_command_line(sys.argv)
	
	
	
	
	
	
	
	
	
	
# notes: 
# Typically forms are used to validate POST and GET content, but they can also be used
# to validate particular values from the URL, or those stored in cookies. 