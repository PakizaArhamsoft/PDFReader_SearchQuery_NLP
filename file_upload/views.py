from django.shortcuts import render
from .classifier import PDFReader
from django.core.files.storage import FileSystemStorage


# Create your views here.
def index(request):
    return render(request, 'file_upload/index.html')


def file_get(request):
    print("______________________")
    try:
        if (request.method == 'POST') and request.FILES['upload']:
            files = []
            d_file = []
            fss = FileSystemStorage()
            for form in request.FILES.getlist('upload'):
                file = fss.save(form.name, form)
                d_file.append(file)
                files.append(fss.url(file))
                print(files)
            query = request.POST['query']
            print("query: ", query)
            pdf_reader = PDFReader()
            doc = pdf_reader.read_pdf(files)
            pos_index = pdf_reader.check_position(doc)
            sample_pos_idx = pdf_reader.SearchQuery(pos_index, query)
            for file in d_file:
                fss.delete(file)
            return render(request, 'file_upload/results.html', {'file_name': files, 'pos_index': pos_index,
                                                                'sample_pos_idx': sample_pos_idx})
    except Exception as e:
        print("Error: ", e)
        return render(request, 'file_upload/results.html')
