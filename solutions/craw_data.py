from newspaper import Article

# Đường dẫn bài viết
url = "https://finance.yahoo.com/news/del-monte-foods-seeks-buyer-012748839.html"

# Tạo đối tượng Article và tải nội dung
article = Article(url)
article.download()
article.parse()

# In tiêu đề, ngày xuất bản và 1000 ký tự đầu tiên
print("Title:", article.title)
print("Publish date:", article.publish_date)
print("Text:", article.text[:1000])
